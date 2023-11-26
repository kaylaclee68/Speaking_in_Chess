import chess
import torch
import numpy as np

def get_rewards(boards: list[chess.Board], dones: list[bool]) -> torch.Tensor:

    targets = []
    loss_masks = []
    for i, board in enumerate(boards):
        result = board.result(claim_draw=True)
        if result == '*':
            if dones[i]:
                num_moves = board.ply() + 1
                targets.append(np.zeros((num_moves,)))

                mask = np.zeros((num_moves,))
                mask[-1] = 1
                loss_masks.append(mask)
            else:
                num_moves = board.ply()
                targets.append(np.zeros((num_moves,)))

                loss_masks.append(np.ones((num_moves,)))

        elif result == '1-0':
            num_moves = board.ply()
            targets.append(np.arange(1, num_moves + 1) % 2)
            loss_masks.append(np.ones((num_moves,)))            
        elif result == '0-1':
            num_moves = board.ply()
            targets.append(np.arange(num_moves) % 2)
            loss_masks.append(np.ones((num_moves,)))
            
        elif result == '1/2-1/2':
            num_moves = board.ply()
            targets.append(np.ones((num_moves,)) / 2)
            loss_masks.append(np.ones((num_moves,)))
        else:
            print("wtf")

    return targets, loss_masks

def get_reverse_discount_rewards(boards: list[chess.Board], discount) -> torch.Tensor:
    growth = 1 - discount
    targets = []
    loss_masks = []
    for board in boards:
        result = board.result(claim_draw=True)
        if result == '*':
            num_moves = board.ply() + 1
            targets.append(np.zeros((num_moves,)))

            mask = np.zeros((num_moves,))
            mask[-1] = 1
            loss_masks.append(mask)

        elif result == '1-0':
            num_moves = board.ply()
            num_w_moves = num_moves // 2 + 1
            w_rew = create_decay_array(num_w_moves, discount)
            b_rew = create_growth_array(num_w_moves - 1, growth)
            targets.append(interleave_arrays(w_rew, b_rew))
            loss_masks.append(np.ones((num_moves,)))            
        elif result == '0-1':
            num_moves = board.ply()
            num_w_moves = num_moves // 2
            b_rew = create_decay_array(num_w_moves, discount)
            w_rew = create_growth_array(num_w_moves, growth)
            targets.append(interleave_arrays(w_rew, b_rew))
            loss_masks.append(np.ones((num_moves,)))
            
        elif result == '1/2-1/2':
            num_moves = board.ply()
            targets.append(np.ones((num_moves,)) / 2)
            loss_masks.append(np.ones((num_moves,)))
        else:
            print("wtf")

    return targets, loss_masks

def pad_arrays(arrays):
    max_length = max(arr.shape[0] for arr in arrays)
    padded_arrays = np.full((len(arrays), max_length), 0.0)
    for i, arr in enumerate(arrays):
        padded_arrays[i, :arr.shape[0]] = arr
    return padded_arrays

def interleave_arrays(a, b):
    # Make sure a is the longer array if they have different lengths
    if len(a) < len(b):
        a, b = b, a

    # Create an empty array of the appropriate size
    c = np.empty(len(a) + len(b), dtype=a.dtype)

    # Fill in the values from a and b
    c[::2] = a
    c[1::2] = b

    return c

def create_decay_array(n, d):
    # Create a reversed array of length n with values [n-1, n-2, ..., 0]
    x = np.arange(n-1, -1, -1)
    
    # Compute the decayed values
    y = 1 - (1 - 0.5) * (1 - d) ** x
    
    return y

def create_growth_array(n, g):
    # Create a reversed array of length n with values [n-1, n-2, ..., 0]
    x = np.arange(n-1, -1, -1)
    
    # Compute the growth values
    y = 0.5 * (1 - g) ** x
    
    return y