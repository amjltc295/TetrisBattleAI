# -*- coding: utf-8 -*-
from engine import TetrisEngine
from heuristic import *
from itertools import count
import numpy as np


width, height = 10, 20 # standard tetris friends rules
engine = TetrisEngine(width, height)




def select_action(engine, shape, anchor, board):
    action_final_location_map = engine.get_valid_final_states(shape, anchor, board)
    act_pairs = [ (k,v[2]) for k,v in action_final_location_map.items()]
    
    placements = [v for k,v in act_pairs]
    h_score = [heuristic_fn(s, complete_line(s)) for s in placements] 
    act_idx = np.argmax(h_score)
    act, placement = act_pairs[act_idx]
    return act, placement

def print_placement(state):
    s = np.asarray(state)
    s = np.swapaxes(s, 1,0)
    print(s)


if __name__ == '__main__':
    # Check if user specified to resume from a checkpoint
    start_epoch = 0
    best_score = float('-inf')
    for i_episode in count(start_epoch):
        # Initialize the environment and state
        state = engine.clear()
        score = 0
        cl = 0
        for t in count():
            # Select and perform an action
            action, placement = select_action(engine, engine.shape, engine.anchor, engine.board)
            # Observations
            state, reward, done, cleared_lines = engine.step_to_final(action)
            # Accumulate reward
            score += reward
            cl += cleared_lines
    
            # Perform one step of the optimization (on the target network)
            if done or t >= 500:
                # Train model
                if i_episode % 1 == 0:
    
                    log = 'epoch {0} score {1} cleared_lines {2}'.format(i_episode, '%.2f' % score, cl)
                    print(log)
                break
    
    
    
    print('Complete')
