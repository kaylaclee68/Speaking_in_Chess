from stockfish import Stockfish
from model import GPTConfig, GPT
from tokenizers import Tokenizer
from chess import pgn
from tqdm import tqdm

import chess
import torch
import argparse
import random
import json
import numpy as np


# model initialization
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
block_size = 768
bias = False # do we use bias inside LayerNorm and Linear layers?
temperature = 0.1

model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size, bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line


def load_model(path, device):
    checkpoint = torch.load(path, map_location=device)
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
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']

    model.to(device)
    checkpoint = None # free up memory

    # compile model?

    return model


def make_move(board, model, tokenizer, num_moves_so_far, turn_list, device):
    turn = num_moves_so_far % 2
    
    turn_list = torch.cat((turn_list, torch.IntTensor([turn]).to(device)))
    idx_next, logit, prob = model.get_next_move(turn_list, board, tokenizer, temperature=temperature)
    
    move = tokenizer.id_to_token(idx_next)

    turn_list = torch.cat((turn_list, idx_next.squeeze(0)))

    return move, turn_list, logit, torch.exp(prob)


def eval(model_1, model_2, tokenizer, rounds, device, max_round, out_path):
    game_log = []
    num_wins = 0
    num_lose = 0
    num_draws = 0

    game_data = []

    for i in tqdm(range(rounds)):
        board = chess.Board()
        game = chess.pgn.Game() # use to record game as pgn

        turn_list = torch.IntTensor([]).to(device)

        if bool(random.getrandbits(1)): # determine who goes first
            first = 0
            game.headers["White"] = "Model_1"
            game.headers["Black"] = "Model_2"
        else:
            first = 1
            game.headers["White"] = "Model_2"
            game.headers["Black"] = "Model_1"

        game.setup(board)
        node = game

        num_moves_so_far = 0

        list_eval = []
        list_logits = []
        list_probs = []

        while (not board.is_game_over()) and num_moves_so_far <= max_round:
            turn = num_moves_so_far % 2
            if turn == first:
                model = model_1
            else:
                model = model_2

            move, turn_list, logit, prob = make_move(board, model, tokenizer, num_moves_so_far, turn_list, device)
            node = node.add_variation(board.parse_san(move))
            list_logits.append(logit.item())
            list_probs.append(prob.item())

            board.push_san(move)

            num_moves_so_far += 1
        
        outcome = board.outcome()

        if outcome:
            result = outcome.result()
            termination = outcome.termination 
        else:
            result = "*"
            termination = None
        
        if termination == chess.Termination.CHECKMATE:            
            if (result == '1-0' and not first) or (result == '0-1' and first):
                num_wins += 1
            else:
                num_lose += 1
        else:
            num_draws += 1

        print(f"*********** Game {i} ***********")
        print(f"Wins: {num_wins}")
        print(f"Loss: {num_lose}")
        print(f"Draw: {num_draws}")

        game.headers["Results"] = board.result()

        print(game, file=open(f"eval_games/game{i}.pgn", "w"), end="\n\n")

        data = dict(
            id=i,
            length=num_moves_so_far,
            evals=list_eval,
            logits=list_logits,
            probs=list_probs
        )

        game_data.append(data)

    
    with open(args.out_path, 'w') as out_file:
        json.dump(game_data, out_file, indent=1)

    return game_log, num_wins, num_lose, num_draws


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-file-1", type=str, required=True)
    parser.add_argument("--checkpoint-file-2", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-plays", type=int, default=50)
    parser.add_argument("--tokenizer-file", type=str, default="../model/tokenizer.model")
    parser.add_argument("--max-round", type=int, default=200)
    parser.add_argument("--out-path", type=str)
    args = parser.parse_args()
    print(args)


    # load checkpoint
    model_1 = load_model(args.checkpoint_file_1, args.device)
    model_2 = load_model(args.checkpoint_file_2, args.device)

    # load tokenizer
    tokenizer = Tokenizer.from_file(args.tokenizer_file)
    tokenizer.enable_truncation(block_size + 1)

    game_log, win, loss, draw = eval(model_1, 
                                     model_2, 
                                     tokenizer, 
                                     args.num_plays, 
                                     args.device, 
                                     args.max_round,
                                     args.out_path)
    
    print(f"Wins: {win}")
    print(f"Loss: {loss}")
    print(f"Draw: {draw}")