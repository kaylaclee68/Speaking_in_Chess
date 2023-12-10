from torch.utils.data import DataLoader
import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from datasets import load_dataset
from tokenizers import Tokenizer

from model import GPTConfig, GPT
from sampler import BatchShuffleSampler
from ccc_dataset import ParquetDataset
from rl_utils import get_rewards, pad_arrays
import chess
import json
import re
import copy

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
epochs = 10
out_dir = 'out_chess_llm_adversarial_with_adv'
eval_interval = 3

save_interval = 5
log_interval = 1
eval_iters = 50
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'resume' # 'scratch' or 'resume' or 'gpt2*'
checkpoint = '/data/evan/CS285_Final_Project/nanoGPT/out_chess_llm/ckpt28000.pt' #'/data/evan/CS285_Final_Project/nanoGPT/out_chess_llm_finetune_2/ckpt18000.pt'
# wandb logging
wandb_log = True # disabled by default
wandb_project = 'lichess'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset_path =  "/data/evan/CS285_Final_Project/data/ccc_processed.parquet" #change to lichess
gradient_accumulation_steps = 64 # used to simulate larger batch sizes
batch_size = 1 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 768
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 2e-5 # max learning rate
max_iters = 70000 # total number of training iterations
weight_decay = 0.0
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 0 # how many steps to warm up for
lr_decay_iters = 70000 # should be ~= max_iters per Chinchilla
min_lr = 2e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16'# if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = False # use PyTorch 2.0 to compile the model to be faster

num_train_steps = 3
num_adversary_steps = 9

update_period = 1500

temperature = 0.1
target_temperature = 0.00001
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------
max_ply = 200

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

tokenizer = Tokenizer.from_file("/data/evan/CS285_Final_Project/model/tokenizer.model")
tokenizer.enable_truncation(block_size + 1)

iter_num = 0
best_val_loss = 1e9

model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size, bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    # if meta_vocab_size is None:
    #     print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = 23296
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {checkpoint}")
    # resume training from a checkpoint.
    ckpt_path = checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = 0 #checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']

adv_model = GPT(gptconf)
adv_model.load_state_dict(model.state_dict())

original_state_dict = copy.deepcopy(model.state_dict())



if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)
adv_model.to(device)

scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
optimizer_adv = adv_model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
# if init_from == 'resume':
#     optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches

def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

openings = ['<w>']


for i in range(max_iters):
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


    # evaluate the loss on train/val sets and write checkpoints
    
        
    if iter_num % save_interval == 0 and master_process:
        if iter_num > 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': model_args,
                'iter_num': iter_num,
                'best_val_loss': 999,
                'config': config,
            }
            print(f"saving checkpoint to {out_dir}")
            torch.save(checkpoint, os.path.join(out_dir, f'ckpt{iter_num}.pt'))
    if iter_num == 0 and eval_only:
        break

    


    optimizer_adv.zero_grad()
    print("Training Adversary...")
    model.requires_grad_(False)
    model.eval()

    adv_model.requires_grad_(True)
    adv_model.train()

    adv_losses_for_logging = []
    adv_ptaus = []
    adv_num_wins = 0
    adv_num_draws = 0
    adv_num_lose = 0
    adv_games_played = 0

    for adv_iter in range(num_adversary_steps):
        # Train on Batch
        for micro_step in range(gradient_accumulation_steps):

            board = chess.Board()

            rand_opening = np.random.choice(openings)

            ids = tokenizer.encode(rand_opening).ids
            
            idx_list = torch.IntTensor(ids).to(device)

            san_list = re.split('<w>|<b>', rand_opening)[1:-1]

            for san in san_list:
                board.push_san(san)
            if board.is_game_over():
                continue
            start_turn = board.turn # true if white          
            ply = 0
            player_picker = ((iter_num + micro_step) % 2) == 0
            players = {chess.WHITE: adv_model if player_picker else model, chess.BLACK: model if player_picker else adv_model}
            model_is_white = player_picker
            q_values_list = []
            probs_list = []
            while not (board.is_game_over() or ply > max_ply):
                
                turn_id = board.turn
                
                is_model = model_is_white == turn_id
                
                temp = 0.1
                
                with ctx:    
                    idx_next, logit, prob = players[turn_id].get_next_move(idx_list, board, tokenizer, temperature=temp)
                    
                
                move = tokenizer.id_to_token(idx_next)
                
                board.push_san(move)
                
                # print(idx_list.shape)
                # print(idx_next.shape)
                
                idx_list = torch.cat((idx_list, idx_next.squeeze(0)))

                idx_list = torch.cat((idx_list, torch.IntTensor([int(not board.turn)]).to(device)))
                
                if is_model:
                    q_values_list.append(logit)
                    probs_list.append(prob)
                
                ply += 1
                
            
            outcome = board.outcome()
            
            if outcome:
                result = outcome.result()
                termination = outcome.termination 
            else:
                result = "*"
                termination = None
                        
            win = None
            
            if result == "*":
                win = 0.5 ** (len(probs_list))
                adv_num_draws += 1
            elif termination == chess.Termination.CHECKMATE:            
                if (result == '1-0' and model_is_white) or (result == '0-1' and not model_is_white):
                    win = 1.0
                    adv_num_wins += 1
                else:
                    win = 0.0
                    adv_num_lose += 1
            elif termination == chess.Termination.THREEFOLD_REPETITION or termination == chess.Termination.FIFTY_MOVES:
                win = 0.0
                adv_num_draws += 1
            else:
                win = 0.5 ** (len(probs_list))
                adv_num_draws += 1

            if len(probs_list) == 0:
                continue
            
            probs = torch.hstack(probs_list).to(device)
            prod_probs = torch.sum(probs)       
            adv_games_played += 1         
            inner_loss = torch.nn.functional.binary_cross_entropy_with_logits(prod_probs.unsqueeze(0), torch.Tensor([win]).to(device)) / gradient_accumulation_steps          
            inner_loss.backward()
            adv_ptaus.append(prod_probs.item())
            adv_losses_for_logging.append(inner_loss.item())

            
        # Eval if log
        if adv_iter % eval_interval == 0 or adv_iter == (num_adversary_steps - 1):
            winrate = adv_num_wins / adv_games_played if adv_games_played > 0 else 0
            drawrate = adv_num_draws / adv_games_played if adv_games_played > 0 else 0
            lossrate = adv_num_lose / adv_games_played if adv_games_played > 0 else 0
            
            mean_loss = np.mean(adv_losses_for_logging) if adv_games_played > 0 else 0

            mean_ptaus = np.mean(adv_ptaus) if adv_games_played > 0 else 0
            
            print(f"step (adversary) {adv_iter}: winrate {winrate} : drawrate {drawrate} : lossrate {lossrate}")
            if wandb_log:
                wandb.log({
                    "global_adv_iter": iter_num * num_adversary_steps + adv_iter,
                    "mean_trajectory_prob": mean_ptaus,
                    "mean_loss": mean_loss,
                    "winrate": winrate,
                    "drawrate": drawrate,
                    "lossrate": lossrate,
                    "lr": lr,
                })
            adv_num_wins = 0
            adv_num_draws = 0
            adv_num_lose = 0
            adv_games_played = 0
            adv_losses_for_logging = []

        optimizer_adv.step()
        
        # Print loss
        if master_process:
            # get loss as float. note: this is a CPU-GPU sync point
            # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
            lossf = inner_loss.item()

            print(f"iter {iter_num}, 'adv iter {adv_iter}: loss {lossf:.9f}")

    # Save space
    optimizer_adv.zero_grad(set_to_none=True)

    if iter_num % save_interval == 0:
        checkpoint = {
            'model': adv_model.state_dict(),
            'optimizer': optimizer_adv.state_dict(),
            'model_args': model_args,
            'iter_num': iter_num,
            'best_val_loss': 999,
            'config': config,
        }
        print(f"saving checkpoint to {out_dir}")
        torch.save(checkpoint, os.path.join(out_dir, f'adv_ckpt{iter_num}.pt'))

    optimizer.zero_grad()
    print("Training Policy...")
    model.requires_grad_(True)
    model.train()

    adv_model.requires_grad_(False)
    adv_model.train()


    train_losses_for_logging = []
    train_ptaus = []
    train_num_wins = 0
    train_num_draws = 0
    train_num_lose = 0
    train_games_played = 0
    for train_iter in range(num_train_steps):
        # Train on Batch
        for micro_step in range(gradient_accumulation_steps):

            board = chess.Board()

            rand_opening = np.random.choice(openings)

            ids = tokenizer.encode(rand_opening).ids
            
            idx_list = torch.IntTensor(ids).to(device)

            san_list = re.split('<w>|<b>', rand_opening)[1:-1]

            for san in san_list:
                board.push_san(san)
            if board.is_game_over():
                continue
            start_turn = board.turn # true if white          
            ply = 0
            player_picker = ((iter_num + micro_step) % 2) == 0
            players = {chess.WHITE: model if player_picker else adv_model, chess.BLACK: adv_model if player_picker else model}
            model_is_white = player_picker
            q_values_list = []
            probs_list = []
            while not (board.is_game_over() or ply > max_ply):
                
                turn_id = board.turn
                
                is_model = model_is_white == turn_id
                
                temp = 0.1
                
                with ctx:    
                    idx_next, logit, prob = players[turn_id].get_next_move(idx_list, board, tokenizer, temperature=temp)
                    
                
                move = tokenizer.id_to_token(idx_next)
                
                board.push_san(move)
                
                # print(idx_list.shape)
                # print(idx_next.shape)
                
                idx_list = torch.cat((idx_list, idx_next.squeeze(0)))

                idx_list = torch.cat((idx_list, torch.IntTensor([int(not board.turn)]).to(device)))
                
                if is_model:
                    q_values_list.append(logit)
                    probs_list.append(prob)
                
                ply += 1
                
            
            outcome = board.outcome()
            
            if outcome:
                result = outcome.result()
                termination = outcome.termination 
            else:
                result = "*"
                termination = None
                        
            win = None
            
            if result == "*":
                win = 0.5 ** (len(probs_list))
                train_num_draws += 1
            elif termination == chess.Termination.CHECKMATE:            
                if (result == '1-0' and model_is_white) or (result == '0-1' and not model_is_white):
                    win = 1.0
                    train_num_wins += 1
                else:
                    win = 0.0
                    train_num_lose += 1
            elif termination == chess.Termination.THREEFOLD_REPETITION or termination == chess.Termination.FIFTY_MOVES:
                win = 0.0
                train_num_draws += 1
            else:
                win = 0.5 ** (len(probs_list))
                train_num_draws += 1

            if len(probs_list) == 0:
                continue
            
            probs = torch.hstack(probs_list).to(device)
            prod_probs = torch.sum(probs)       
            train_games_played += 1         
            inner_loss = torch.nn.functional.binary_cross_entropy_with_logits(prod_probs.unsqueeze(0), torch.Tensor([win]).to(device)) / gradient_accumulation_steps          
            inner_loss.backward()
            train_ptaus.append(prod_probs.item())
            train_losses_for_logging.append(inner_loss.item())

            
        # Eval if log
        if train_iter % eval_interval == 0 or train_iter == (num_train_steps - 1):
            winrate = train_num_wins / train_games_played if train_games_played > 0 else 0
            drawrate = train_num_draws / train_games_played if train_games_played > 0 else 0
            lossrate = train_num_lose / train_games_played if train_games_played > 0 else 0
            
            mean_loss = np.mean(train_losses_for_logging) if train_games_played > 0 else 0

            mean_ptaus = np.mean(train_ptaus) if train_games_played > 0 else 0
            
            print(f"step (train) {train_iter}: winrate {winrate} : drawrate {drawrate} : lossrate {lossrate}")
            if wandb_log:
                wandb.log({
                    "global_train_iter": iter_num * num_train_steps + train_iter,
                    "mean_trajectory_prob": mean_ptaus,
                    "mean_loss": mean_loss,
                    "winrate": winrate,
                    "drawrate": drawrate,
                    "lossrate": lossrate,
                    "lr": lr,
                })
            train_num_wins = 0
            train_num_draws = 0
            train_num_lose = 0
            train_games_played = 0
            train_losses_for_logging = []

        optimizer.step()
        
        # Print loss
        if master_process:
            # get loss as float. note: this is a CPU-GPU sync point
            # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
            lossf = inner_loss.item()

            print(f"iter {iter_num}, 'train iter {train_iter}: loss {lossf:.9f}")
    

    print("Reset Adv Model")
    adv_model.load_state_dict(original_state_dict)

    iter_num += 1
    # run loss on trajectory

    # backprop

    # collect metrics

