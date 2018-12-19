from copy import deepcopy
import random

import numpy as np

shapes = {
    'T': [(0, 0), (-1, 0), (1, 0), (0, -1)],
    'J': [(0, 0), (-1, 0), (0, -1), (0, -2)],
    'L': [(0, 0), (1, 0), (0, -1), (0, -2)],
    'S': [(0, 0), (-1, 0), (0, -1), (1, -1)],
    'Z': [(0, 0), (-1, -1), (0, -1), (1, 0)],
    'I': [(0, 0), (0, -1), (0, -2), (0, -3)],
    'O': [(0, 0), (0, -1), (-1, 0), (-1, -1)],
}
shape_names = ['T', 'J', 'L', 'Z', 'S', 'I', 'O']


def rotated(shape, cclk=False):
    if cclk:
        return [(-j, i) for i, j in shape]
    else:
        return [(j, -i) for i, j in shape]


def is_occupied(shape, anchor, board):
    for i, j in shape:
        x, y = anchor[0] + i, anchor[1] + j
        if y < 0:
            continue
        if x < 0 or x >= board.shape[0] or y >= board.shape[1] or board[x, y]:
            return True
    return False


def left(shape, anchor, board):
    new_anchor = (anchor[0] - 1, anchor[1])
    return (shape, anchor) if is_occupied(shape, new_anchor, board) else (shape, new_anchor)


def right(shape, anchor, board):
    new_anchor = (anchor[0] + 1, anchor[1])
    return (shape, anchor) if is_occupied(shape, new_anchor, board) else (shape, new_anchor)


def soft_drop(shape, anchor, board):
    new_anchor = (anchor[0], anchor[1] + 1)
    return (shape, anchor) if is_occupied(shape, new_anchor, board) else (shape, new_anchor)


def hard_drop(shape, anchor, board):
    while True:
        _, anchor_new = soft_drop(shape, anchor, board)
        if anchor_new == anchor:
            return shape, anchor_new
        anchor = anchor_new


def rotate_left(shape, anchor, board):
    new_shape = rotated(shape, cclk=False)
    return (shape, anchor) if is_occupied(new_shape, anchor, board) else (new_shape, anchor)


def rotate_right(shape, anchor, board):
    new_shape = rotated(shape, cclk=True)
    return (shape, anchor) if is_occupied(new_shape, anchor, board) else (new_shape, anchor)


def idle(shape, anchor, board):
    return (shape, anchor)


class TetrisEngine:
    def __init__(self, width, height, enable_KO=True):
        self.width = width
        self.height = height
        self.board = np.zeros(shape=(width, height), dtype=np.float)

        # actions are triggered by letters
        self.value_action_map = {
            "move_left": left,
            "move_right": right,
            "hard_drop": hard_drop,
            "soft_drop": soft_drop,
            "rotate_left": rotate_left,
            "rotate_right": rotate_right,
            "idle": idle,
            "hold": self.hold,
        }
        self.action_value_map = dict([(j, i) for i, j in self.value_action_map.items()])
        self.nb_actions = len(self.value_action_map)

        # for running the engine
        self.time = -1
        self.score = -1
        self.anchor = None
        self.shape = None
        self.shape_name = None
        self.n_deaths = 0
        self.game_over = False

        # used for generating shapes
        self._shape_counts = [0] * len(shapes)

        # states
        self.total_cleared_lines = 0
        self.previous_garbage_lines = 0
        self.garbage_lines = 0
        self.highest_line = 0
        self.drop_count = 0
        self.step_num_to_drop = 30
        self.hold_locked = False
        self.hold_shape = []
        self.hold_shape_name = None
        self.next_shape_name, self.next_shape = self._choose_shape()

        self.enable_KO = enable_KO  # clear only the garbage lines after dead

        # clear after initializing
        self.clear()

    def _choose_shape(self):
        maxm = max(self._shape_counts)
        m = [5 + maxm - x for x in self._shape_counts]
        r = random.randint(1, sum(m))
        for i, n in enumerate(m):
            r -= n
            if r <= 0:
                self._shape_counts[i] += 1
                return shape_names[i], shapes[shape_names[i]]

    def _new_piece(self):
        self.shape_name, self.shape = self.next_shape_name, self.next_shape
        self.anchor = (self.width // 2, 1)
        while is_occupied(self.shape, (self.anchor[0], self.anchor[1]), self.board):
            self.anchor = (self.anchor[0], self.anchor[1] - 1)

        self.next_shape_name, self.next_shape = self._choose_shape()

    def _has_dropped(self):
        return is_occupied(self.shape, (self.anchor[0], self.anchor[1] + 1), self.board)

    def clear_lines(self, board):
        can_clear = [True if sum(board[:, i]) == self.width else False for i in range(self.height)]
        new_board = np.zeros_like(board)
        j = self.height - 1
        for i in range(self.height - 1, -1, -1):
            if not can_clear[i]:
                new_board[:, j] = board[:, i]
                j -= 1

        return sum(can_clear), new_board

    def valid_action_count(self):
        valid_action_sum = 0

        for value, fn in self.value_action_map.items():
            if value == "hold":
                continue
            # If they're equal, it is not a valid action
            if fn(self.shape, self.anchor, self.board) != (self.shape, self.anchor):
                valid_action_sum += 1

        return valid_action_sum

    def step(self, action):
        self.shape, self.anchor = self.value_action_map[action](self.shape, self.anchor, self.board)

        reward = self.valid_action_count()

        # Drop each step_num_to_drop step
        cleared_lines = 0
        if self.drop_count == self.step_num_to_drop or action == "hard_drop":
            self.drop_count = 0
            if action != "soft_drop":
                self.shape, self.anchor = soft_drop(self.shape, self.anchor, self.board)
            if self._has_dropped():
                cleared_lines, KOed = self._handle_dropped(reward)
                reward += cleared_lines * 10
                reward -= KOed * 10

        # Update time and reward
        self.time += 1
        self.drop_count += 1

        self.board = self.set_piece(self.shape, self.anchor, self.board, True)
        state = np.copy(self.board)
        self.board = self.set_piece(self.shape, self.anchor, self.board, False)
        self._update_states()
        return state, reward, self.game_over, cleared_lines

    def _handle_dropped(self, done=False):
        self.board = self.set_piece(self.shape, self.anchor, self.board, True)
        cleared_lines, self.board = self.clear_lines(self.board)
        self.score += cleared_lines
        self.total_cleared_lines += cleared_lines
        KOed = False
        if (np.any(self.board[:, 0])):
            self.board = self.set_piece(self.shape, self.anchor, self.board, True)
            if self.garbage_lines == 0:
                self.game_over = True
            self.clear()
            self.n_deaths += 1
            KOed = True
        else:
            self._new_piece()
            self.hold_locked = False
        return cleared_lines, KOed

    def step_to_final(self, action):
        reward = 0

        # actions that directly go to the final locations
        action_final_location_map = self.get_valid_final_states(
            self.shape, self.anchor, self.board)
        self.shape, self.anchor, self.board, actions = action_final_location_map[action]
        cleared_lines, reward, done = self._handle_dropped(reward)
        self.board = self.set_piece(self.shape, self.anchor, self.board, True)
        state = np.copy(self.board)
        self.board = self.set_piece(self.shape, self.anchor, self.board, False)
        self._update_states()

        return state, reward, done, cleared_lines

    def clear(self):
        if not self.enable_KO:
            self.time = 0
            self.score = 0
            self.board = np.zeros_like(self.board)
        self._new_piece()
        self.hold_locked = False
        self.garbage_lines = 0
        self.highest_line = 0

        return self.board

    def set_piece(self, shape, anchor, board, on=False):
        new_board = deepcopy(board)
        for i, j in shape:
            x, y = i + anchor[0], j + anchor[1]
            if x < self.width and x >= 0 and y < self.height and y >= 0:
                new_board[anchor[0] + i, anchor[1] + j] = on
        return new_board

    def __repr__(self):
        self.board = self.set_piece(self.shape, self.anchor, self.board, True)
        s = f"Hold: {self.hold_shape_name}\n"
        s += f"Next: {self.next_shape_name}\n"
        s += 'o' + '-' * self.width + 'o'
        for line in self.board.T[1:]:
            display_line = ['\n|']
            for grid in line:
                if grid == -1:
                    display_line.append('X')
                elif grid:
                    display_line.append('O')
                else:
                    display_line.append(' ')
            display_line.append('|')
            s += "".join(display_line)

        s += '\no' + '-' * self.width + 'o\n'
        self.board = self.set_piece(self.shape, self.anchor, self.board, False)
        return s

    def receive_garbage_lines(self, garbage_lines):
        self.garbage_lines += garbage_lines

    def is_alive(self):
        if self.highest_line >= self.height:
            return False
        return True

    def _update_states(self):
        new_board = np.zeros_like(self.board)
        if self.garbage_lines > 0:
            new_board[:, -self.garbage_lines:] = -1
        for i in range(self.height - self.previous_garbage_lines - 1, -1, -1):
            new_board[:, i - (self.garbage_lines - self.previous_garbage_lines)] = self.board[:, i]

        while is_occupied(self.shape, self.anchor, new_board):
            self.anchor = (self.anchor[0], self.anchor[1] - 1)

        self.previous_garbage_lines = self.garbage_lines
        self.board = new_board
        for i in range(self.height - 1, -1, -1):
            if sum(self.board[:, i]) > 0:
                self.highest_line = self.height - i

    def hold(self, shape, anchor, board):
        if self.hold_locked:
            return (shape, anchor)
        else:
            self.hold_locked = True
            tmp_shape_name = self.shape_name
            if len(self.hold_shape) == 0:
                self._new_piece()
            else:
                self.shape = self.hold_shape
                self.shape_name = self.hold_shape_name
            self.hold_shape = shape
            self.hold_shape_name = tmp_shape_name

        # Prevent collision after hold
        actions = ["move_left", "move_right"]
        count = -1
        while is_occupied(self.shape, self.anchor, board):
            count += 1
            try:
                action = actions[count]
                self.shape, self.anchor = self.value_action_map[action](self.shape, self.anchor, board)
            except Exception:
                self.anchor = (self.anchor[0], self.anchor[1] - 1)
        return self.shape, self.anchor

    def get_valid_final_states(self, shape, anchor, board):
        # Reference https://github.com/brendanberg01/TetrisAI/blob/master/ai.py
        action_state_dict = {}
        for move in range(-self.width // 2, self.width // 2 + 1):
            for rotate in range(0, 4):
                actions = []
                final_shape, final_anchor, final_board = shape, anchor, deepcopy(board)
                for i in range(rotate):
                    actions.append("rotate_right")  # right_rotate
                if move > 0:
                    for i in range(move):
                        actions.append("move_right")  # right
                else:
                    for i in range(-move):
                        actions.append("move_left")  # left

                actions.append("hard_drop")  # hard_drop
                for action in actions:
                    final_shape, final_anchor = self.value_action_map[action](
                        final_shape, final_anchor, board
                    )
                final_board = self.set_piece(final_shape, final_anchor, board, True)
                action_name = f"move_{move}_right_rotate_{rotate}"
                action_state_dict[action_name] = (final_shape, final_anchor, final_board, actions)
        return action_state_dict

    def get_board(self):
        self.board = self.set_piece(self.shape, self.anchor, self.board, True)
        state = np.copy(self.board)
        self.board = self.set_piece(self.shape, self.anchor, self.board, False)
        return state
