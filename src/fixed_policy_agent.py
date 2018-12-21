# -*- coding: utf-8 -*-
from engine import TetrisEngine
from heuristic import heuristic_fn, complete_line
from itertools import count
import numpy as np


width, height = 10, 20  # standard tetris friends rules
engine = TetrisEngine(width, height, enable_KO=False)


class FixedPolicyAgent:
    def __init__(self):
        self.current_actions = []

    def get_action(self, engine, shape, anchor, board):
        if len(self.current_actions) == 0:
            _, _, self.current_actions = self.select_action(engine, shape, anchor, board)
        action = self.current_actions.pop(0)
        return action

    def select_action(self, engine, shape, anchor, board):
        actions_name_final_location_map = engine.get_valid_final_states(shape, anchor, board)
        act_pairs = [(k, v[2], v[3]) for k, v in actions_name_final_location_map.items()]
        placements = [p for k, p, actions in act_pairs]
        h_score = [heuristic_fn(s, complete_line(s)) for s in placements]
        act_idx = np.argmax(h_score)
        actions_name, final_placement, actions = act_pairs[act_idx]
        return actions_name, final_placement, actions


def print_placement(state):
    s = np.asarray(state)
    s = np.swapaxes(s, 1, 0)
    print(s)


agent = FixedPolicyAgent()
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
            actions_name, placement, actions = agent.select_action(engine, engine.shape, engine.anchor, engine.board)
            # Observations
            state, reward, done, cleared_lines = engine.step_to_final(actions_name)
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
