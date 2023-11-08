from datasets import load_dataset
import pandas as pd
import numpy as np
import chess.pgn
import chess
import re
from hashlib import sha256
import multiprocessing
import argparse
import os
import jsonlines

def process_game(game: chess.pgn.Game):
    board = game.board()
    seq = ""
    turns = ['<b>', '<w>']

    for move in game.mainline_moves():
        bool_turn = board.turn
        turn = turns[int(bool_turn)]
        lan = board.lan(move)
        seq += turn + lan
        board.push(move)

    return seq

def pool_process(inputs):
    dirname, fname, outdir = inputs
    pgn = open(os.path.join(dirname, fname), encoding="utf-8")
    game = chess.pgn.read_game(pgn)
    while game:
        try:
            id = sha256((str(game.headers) + str(game.mainline())).encode('utf-8')).hexdigest()
            result = game.headers.get("Result")
            
            game_seq = process_game(game)
            
            with jsonlines.open(os.path.join(outdir, f"{os.path.splitext(fname)[0]}.jsonl"), 'a') as writer:
                writer.write(dict(id=id, game=game_seq, result=result))
        except Exception as e:
            print(f"ERROR in {fname}")
            print(e)
            
        game = chess.pgn.read_game(pgn)
        
    return
    

def fname_iter(dirname, outdir):
    for fname in os.listdir(dirname):
        if os.path.splitext(fname)[1] == ".pgn":
            yield dirname, fname, outdir
    return

def main(args):
    os.mkdir(args.outdir)
                
    print(f"Spawning {args.num_processes} processes...")
    with multiprocessing.Pool(processes=args.num_processes) as pool:
        pool.map(pool_process, fname_iter(args.dirname, args.outdir), chunksize=1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirname", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument('--num-processes', type=int, default=64, help='Number of parallel process to spawn')
    
    args = parser.parse_args()
    
    main(args)
    