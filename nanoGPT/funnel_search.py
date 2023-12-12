from model import GPTConfig, GPT
from tokenizers import Tokenizer
from contextlib import nullcontext
from chess import pgn
from tqdm import tqdm
from pytorch_forecasting import utils

import chess
import torch
import torch.nn as nn
import argparse
import random
import numpy as np


PAD_TOKEN = 23253


def load_model(path, device):
    checkpoint = torch.load(path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']

    # create the model
    gptconf = GPTConfig(**checkpoint_model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']

    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']

    model.to(device)
    checkpoint = None # free up memory

    return model, checkpoint_model_args, iter_num, best_val_loss


def save_games(trajs, tokenizer, num_games):
    # indices = np.random.choice(len(trajs), num_games, replace=False)
    indices = np.arange(len(trajs))
    print(indices)
    for count, i in enumerate(indices):
        traj = trajs[i][0]

        moves = [tokenizer.id_to_token(id) for id in traj[1::2]]

        print(moves[:5], moves[-5:], len(moves))

        game = chess.pgn.Game()
        board = chess.Board()

        game.setup(board)
        node = game
        for move in moves:
            node = node.add_variation(board.parse_san(move))
            board.push_san(move)

        print(game, file=open(f"funnel_games/game{count}.pgn", "w"), end="\n\n")


def collect_funnel(model: GPT, 
                   tokenizer: Tokenizer,
                   k_seqs: list,
                   m_seqs: list, 
                   temperature: float, 
                   device: str, 
                   max_round: int):
    board = chess.Board()
    idx_list = torch.IntTensor([]).to(device)
    trajectories = model.funnel_search(idx_list,
                                       k_seqs,
                                       m_seqs,
                                       tokenizer,
                                       board,
                                       temperature,
                                       device,
                                       1,
                                       max_round)
    
    return trajectories


def train(model: GPT, 
          tokenizer: Tokenizer,
          k_seqs: list,
          m_seqs: list,
          iterations: int, 
          temperature: float, 
          device: str,
          batch_size: int,
          max_round: int,
          eval_freq: int):
    # dtype = 'bfloat16'
    # ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    # ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type=device, dtype=ptdtype)
    
    for i in tqdm(range(iterations)):

        # train
        # with ctx:
        trajectories = collect_funnel(model,
                                      tokenizer,
                                      k_seqs,
                                      m_seqs,
                                      temperature,
                                      device,
                                      max_round)
        
        for i in range(0, len(trajectories), batch_size):
            batch = trajectories[i:i+batch_size]
            X = torch.IntTensor([]).to(device)
            Y = torch.IntTensor([]).to(device)

            for idx, board, _ in batch:
                print(X.size())
                pad_length = max(X.size()[-1], idx.size()[0])
                X = torch.cat((nn.functional.pad(X, (0, pad_length), "constant", PAD_TOKEN), 
                               nn.functional.pad(idx.unsqueeze(0), (0, pad_length), "constant", PAD_TOKEN)), 
                               dim=0)

                outcome = board.outcome().result()
                result = torch.IntTensor([int(outcome[0]), int(outcome[2])]).to(device)
                Y = torch.cat((Y, result.unsqueeze(0)), dim=0)
        
            print("***********************")
            print(X)
            print("***********************")
            print(Y)
            _, loss = model.forward(idx=X, targets=Y, offline=True)

        print(loss)
        # # validation
        # eval = i % eval_freq == 0 # record game every pgn-freq step
        # if eval:
        #     game = chess.pgn.Game()
        #     game.setup(board)
        #     node = game


    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-file", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--tokenizer-file", type=str, default="../model/tokenizer.model")
    parser.add_argument("--num-iter", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-round", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--eval-freq", type=int, default=100)
    parser.add_argument("--num-save-games", type=int, default=0)
    args = parser.parse_args()
    print(args)

    # load checkpoint
    model, checkpoint_args, iters, best_loss = load_model(args.checkpoint_file, args.device)

    print("********** Checkpoint Model Stats **********")
    print(f"Iterations: {iters}")
    print(f"Best Validation Loss: {best_loss}")

    # load tokenizer
    tokenizer = Tokenizer.from_file(args.tokenizer_file)
    tokenizer.enable_truncation(checkpoint_args["block_size"] + 1)

    # create a k sequence
    k_seqs = [2 for i in range(200)]
    m_seqs = []
    # for i in range(20):
    #     m_seqs.extend([(i + 1) ** 2] * 10) # this k seqs double k every 10 moves
    for i in range(200):
        m_seqs.extend([10])

    # trajs = collect_funnel(model, tokenizer, k_seqs, m_seqs, args.temperature, args.device, args.max_round)

    # print(f"**** Number of trajectories collected: {len(trajs)} ****")
    # save_games(trajs, tokenizer, args.num_save_games)
    

    train(model,
          tokenizer,
          k_seqs,
          m_seqs,
          args.num_iter, 
          args.temperature, 
          args.device, 
          args.batch_size,
          args.max_round,
          args.eval_freq)