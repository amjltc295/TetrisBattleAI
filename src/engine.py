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
value_to_shape_names_map = {
    1: 'T',
    2: 'J',
    3: 'L',
    4: 'Z',
    5: 'S',
    6: 'I',
    7: 'O'
}
shape_name_to_value_map = dict([(j, i) for i, j in value_to_shape_names_map.items()])


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


def move_left(shape, anchor, board):
    new_anchor = (anchor[0] - 1, anchor[1])
    return (shape, anchor) if is_occupied(shape, new_anchor, board) else (shape, new_anchor)


def move_right(shape, anchor, board):
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


def combo_to_line_sent(combo):
    combo_line_sent_map = {
        -1: 0,
        0: 0,
        1: 1,
        2: 1,
        3: 2,
        4: 2,
        5: 3,
        6: 3,
        7: 4,
    }

    if combo <= 7:
        return combo_line_sent_map[combo]
    else:
        return 4


def board_to_bool(board):
    board_bool = np.zeros_like(board)
    board_bool[board > 0] = True
    board_bool[board <= 0] = False
    return board_bool


class TetrisEngine:
    def __init__(self, width, height, enable_KO=True):
        self.width = width
        self.height = height
        self.board = np.zeros(shape=(width, height), dtype=np.float)

        # actions are triggered by letters
        self.value_action_map = {
            "move_left": move_left,
            "move_right": move_right,
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

        # used for generating shapes
        self._shape_counts = [0] * len(shapes)

        # states
        self.total_cleared_lines = 0
        self.total_sent_lines = 0
        self.combo = -1
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
                return value_to_shape_names_map[i+1], shapes[value_to_shape_names_map[i+1]]

    def _new_piece(self):
        self.shape_name, self.shape = self.next_shape_name, self.next_shape
        self.anchor = (self.width // 2, 1)
        while is_occupied(self.shape, self.anchor, self.board):
            self.anchor = (self.anchor[0], self.anchor[1] - 1)

        self.next_shape_name, self.next_shape = self._choose_shape()

    def _has_dropped(self):
        return is_occupied(self.shape, (self.anchor[0], self.anchor[1] + 1), self.board)

    def clear_lines(self, board):
        board_bool = board_to_bool(board)
        can_clear = [True if sum(board_bool[:, i]) == self.width else False for i in range(self.height)]
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
        game_over = False
        sent_lines = 0
        if self.drop_count == self.step_num_to_drop or action == "hard_drop":
            self.drop_count = 0
            if action != "soft_drop":
                self.shape, self.anchor = soft_drop(self.shape, self.anchor, self.board)
            if self._has_dropped():
                cleared_lines, KOed, game_over, sent_lines = self._handle_dropped()
                reward += cleared_lines * 10
                reward -= (KOed if self.enable_KO else game_over) * 100

        # Update time and reward
        self.time += 1
        self.drop_count += 1

        state = self.get_board()
        if not game_over:
            game_over = self._update_states()
        return state, reward, game_over, cleared_lines, sent_lines

    def _handle_dropped(self):
        self.board = self.set_piece(self.shape, self.anchor, self.board, True)
        cleared_lines, self.board = self.clear_lines(self.board)
        self.score += cleared_lines
        self.total_cleared_lines += cleared_lines
        self.combo = self.combo + cleared_lines if cleared_lines > 0 else -1
        sent_lines = combo_to_line_sent(self.combo)
        self.total_sent_lines += sent_lines
        KOed = False
        game_over = False
        if np.any(board_to_bool(self.board)[:, 0]):
            self.board = self.set_piece(self.shape, self.anchor, self.board, True)
            if self.garbage_lines == 0:
                game_over = True
            self.clear()
            self.n_deaths += 1
            KOed = True
        else:
            self._new_piece()
            self.hold_locked = False
        return cleared_lines, KOed, game_over, sent_lines

    def step_to_final(self, actions):
        # actions: list of actions that directly go to the final locations
        for action in actions:
            state, reward, game_over, cleared_lines, sent_lines = self.step(action)

        return state, reward, game_over, cleared_lines, sent_lines

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

    def set_piece(self, shape, anchor, board, on=False, shape_value=None):
        if shape_value is None:
            shape_value = shape_name_to_value_map[self.shape_name]
        if on:
            on = shape_value
        new_board = deepcopy(board)
        for i, j in shape:
            x, y = i + anchor[0], j + anchor[1]
            if x < self.width and x >= 0 and y < self.height and y >= 0:
                new_board[anchor[0] + i, anchor[1] + j] = on
        return new_board

    def __repr__(self):
        board = self.get_board()
        s = f"Hold: {self.hold_shape_name}  Next: {self.next_shape_name}\n"
        s += 'o' + '-' * self.width + 'o'
        for line in board.T[1:]:
            display_line = ['\n|']
            for grid in line:
                if grid == -1:
                    display_line.append('X')
                elif grid == -2:
                    display_line.append('V')
                elif grid:
                    display_line.append('O')
                else:
                    display_line.append(' ')
            display_line.append('|')
            s += "".join(display_line)

        s += '\no' + '-' * self.width + 'o\n'
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

        # Check if additional garbage_lines squeeze the floating block
        # If so, force it to drop
        squeezed = 0
        while is_occupied(self.shape, self.anchor, new_board):
            self.anchor = (self.anchor[0], self.anchor[1] - 1)
            squeezed += 1
        if squeezed > 1:
            self.shape, self.anchor = hard_drop(self.shape, self.anchor, new_board)
            new_board = self.set_piece(self.shape, self.anchor, new_board, True)

        # Check additional garbage_lines cause KO
        game_over = False
        if np.any(board_to_bool(new_board)[:, 0]):
            if self.garbage_lines == 0:
                game_over = True
            new_board = np.zeros_like(self.board)
            for i in range(self.height - self.previous_garbage_lines - 1, -1, -1):
                new_board[:, i + self.previous_garbage_lines] = self.board[:, i]
            new_board = self.set_piece(self.shape, self.anchor, new_board, False)
            self.shape, self.anchor = hard_drop(self.shape, self.anchor, new_board)
            new_board = self.set_piece(self.shape, self.anchor, new_board, True)
            self.clear()
            self.n_deaths += 1

        self.previous_garbage_lines = self.garbage_lines
        self.board = new_board
        for i in range(self.height - 1, -1, -1):
            if sum(self.board[:, i]) > 0:
                self.highest_line = self.height - i
        return game_over

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
                count = -1
        return self.shape, self.anchor

    def get_valid_final_states(self, shape, anchor, board, enable_hold=True):
        # Reference https://github.com/brendanberg01/TetrisAI/blob/master/ai.py
        action_state_dict = {}
        candidate_shapes = {
            0: shape,
            1: self.hold_shape if self.hold_shape_name is not None else self.next_shape
        }
        for hold, chosen_shape in candidate_shapes.items():
            if not enable_hold and hold:
                break
            for move in range(-self.width // 2, self.width // 2 + 1):
                for rotate in range(0, 4):
                    actions = []
                    final_shape, final_anchor, final_board = chosen_shape, anchor, deepcopy(board)
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
                    if hold:
                        actions.insert(0, 'hold')
                    final_board = self.set_piece(final_shape, final_anchor, board, True)
                    action_name = f"move_{move}_right_rotate_{rotate}_hold_{hold}"
                    action_state_dict[action_name] = (final_shape, final_anchor, final_board, actions)
        return action_state_dict

    def get_board(self):
        shape, anchor = hard_drop(self.shape, self.anchor, self.board)
        hard_dropped_board = self.set_piece(shape, anchor, self.board, True, -2)
        board = self.set_piece(self.shape, self.anchor, hard_dropped_board, True)
        return board
