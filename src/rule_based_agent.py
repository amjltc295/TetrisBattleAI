# -*- coding: utf-8 -*-
import numpy as np

from engine import TetrisEngine, board_to_bool
from heuristic import heuristic_fn2, complete_line, get_height

width, height = 10, 20  # standard tetris friends rules
engine = TetrisEngine(width, height, enable_KO=False)


class RuleBasedAgent:
    def __init__(self):
        self.current_actions = []

    def get_action(self, engine, shape, anchor, board):
        if len(self.current_actions) == 0:
            _, _, self.current_actions = self.select_action(engine, shape, anchor, board)
        action = self.current_actions.pop(0)
        return action

    def select_action(self, engine, shape, anchor, board):
        actions_name_final_location_map = engine.get_valid_final_states(shape, anchor, board)
        act_pairs = [(k, board_to_bool(v[2]), v[3]) for k, v in actions_name_final_location_map.items()]
        placements = [p for k, p, actions in act_pairs]
        combo = engine.combo > -1
        h_score = [heuristic_fn2(s, complete_line(s), combo) for s in placements]
        if engine.shape_name == 'I':
            if max(h_score) < 1000:
                for i in range(44):
                    h_score[i] = -20000
        act_idx = np.argmax(h_score)
        actions_name, final_placement, actions = act_pairs[act_idx]
        return actions_name, final_placement, actions
