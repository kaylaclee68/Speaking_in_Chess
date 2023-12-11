from model import GPTConfig, GPT
from tokenizers import Tokenizer
from chess import pgn
from tqdm import tqdm

import chess
import torch
import argparse
import random


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
          iterations: int, 
          temperature: float, 
          device: str, 
          max_round: int,
          pgn_freq: int):
    
    for i in tqdm(iterations):
        board = chess.Board()

        # train
        model.funnel_search(torch.Tensor([]).to(device),
                            k_seqs,
                            tokenizer,
                            board,
                            temperature,
                            m)

        # validation
        record = i % pgn_freq == 0 # record game every pgn-freq step
        if record:
            game = chess.pgn.Game()
            game.setup(board)
            node = game

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-file", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--tokenizer-file", type=str, default="../model/tokenizer.model")
    # parser.add_argument("--num-iter", type=int, required=True)
    parser.add_argument("--max-round", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--pgn-freq", type=int, default=100)
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
    k_seqs = [5 for i in range(200)]
    m_seqs = []
    for i in range(20):
        m_seqs.extend([(i + 1) ** 2] * 10) # this k seqs increment k by 1 every 5 moves

    trajs = collect_funnel(model, tokenizer, k_seqs, m_seqs, args.temperature, args.device, args.max_round)

    print(len(trajs))

    # train(model,
    #       tokenizer,
    #       args.k_seqs,
    #       args.num_iter, 
    #       args.temperature, 
    #       args.device, 
    #       args.max_round,
    #       args.pgn_freq)