# -*- coding: utf-8 -*-
import random


class RandomActionAgent:
    def __init__(self):
        self.actions = [
            "move_left",
            "move_right",
            "hard_drop",
            "soft_drop",
            "rotate_left",
            "rotate_right",
            "idle",
            "hold"
        ]

    def get_action(self, engine, shape, anchor, board):
        return random.choice(self.actions)


random_agent = RandomActionAgent()
