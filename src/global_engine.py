import argparse
import curses
import time

from engine import TetrisEngine
from gui.gui import GUI


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ww', '--width', type=int, default=10, help='Window width')
    parser.add_argument('-hh', '--height', type=int, default=16, help='Window height')
    parser.add_argument('-n', '--player_num', type=int, default=2, help='Number of players')
    parser.add_argument('-b', '--block_size', type=int, default=30, help='Set block size to enlarge GUI')
    parser.add_argument('-g', '--use_gui', type=int, default=0, help='Active output to gui')
    parser.add_argument('-f', '--step_to_final', default=False, action='store_true',
                        help='One step to the final location')
    args = parser.parse_args()
    return args


class GlobalEngine:
    def __init__(
        self, width, height, player_num, use_gui, block_size,
        game_time=120, KO_num_to_win=2
    ):
        self.width = width
        self.height = height
        self.player_num = player_num
        self.game_time = game_time
        self.KO_num_to_win = KO_num_to_win
        self.winner = None

        # For GUI use
        self.gui = None
        self.block_size = block_size
        self.use_gui = use_gui
        self.pause = False

        self.engines = {}
        for i in range(player_num):
            self.engines[i] = TetrisEngine(width, height)
            self.engines[i].clear()

        self.global_state = {}
        self.engine_states = {}

    def setup(self):
        # Initialization
        if self.use_gui:
            gui = GUI(self, self.block_size)
            self.gui = gui
        else:
            self.stdscr = curses.initscr()

        # Store play information
        self.dbs = {}

        self.done = False

        for i in range(self.player_num):
            # Initial rendering
            if not self.use_gui:
                self.stdscr.addstr(str(self.engines[i]))
            self.engine_states[i] = {
                "KO": 0,
                "reward": 0,
                "lines_sent": 0,
                "hold_shape": None,
                "hold_shape_name": None,
                "hold_locked": False,
                "garbage_lines": 0,
                "highest_line": 0
            }
            self.key_action_map = {
                ord('a'): 0,  # Shift left
                ord('d'): 1,  # Shift right
                ord('w'): 2,  # Hard drop
                ord('s'): 3,  # Soft drop
                ord('q'): 4,  # Rotate left
                ord('e'): 5,  # Rotate right
                ord('f'): 7   # Hold
            }
            # Initialize dbs
            self.dbs[i] = []

        self.start_time = time.time()

    def keyboard_control(self, key):
        if key in self.key_action_map:
            action = self.key_action_map[key]
        else:
            action = 6
        return action

    def sent_lines(self, idx, cleared_lines):
        for other_idx, other_engine in self.engines.items():
            if other_idx != idx:
                other_engine.receive_garbage_lines(cleared_lines)

                # Get KO
                if self.player_num == 2:
                    self.engine_states[idx]['KO'] = other_engine.n_deaths
                elif cleared_lines > 0:
                    if not other_engine.is_alive():
                        self.engine_states[idx]['KO'] += 1

    def compare_score(self):
        max_score = 0
        for idx, engine in self.engines.items():
            score = self.engine_states[idx]['KO'] * 10000 + self.engine_states[idx]['lines_sent']
            if score > max_score:
                self.winner = idx
                max_score = score

    def get_action(self, step_to_final):
        if self.use_gui:
            key = self.gui.last_gui_input()
            """
            while key not in self.key_action_map:
                key = self.gui.last_gui_input()
                self.gui.update_screen()

            """
        else:
            key = self.stdscr.getch()
        if step_to_final:
            move = chr(key)
            if move == '-':
                move += chr(key)
            rotate = chr(key)
            action = f"move_{move}_right_rotate_{rotate}"
        else:
            action = self.keyboard_control(key)
        return action

    def play_game(self, step_to_final=False):
        # Initialization
        self.setup()

        game_over = False
        while time.time() - self.start_time < self.game_time and not game_over:
            action = self.get_action(step_to_final)

            if not self.use_gui:
                self.stdscr.clear()
            for idx, engine in self.engines.items():
                # Game step
                if step_to_final:
                    state, reward, self.done, cleared_lines = engine.step_to_final(action)
                else:
                    state, reward, self.done, cleared_lines = engine.step(action)

                # Update state
                self.set_engine_state(idx, engine, reward, cleared_lines)
                self.sent_lines(idx, cleared_lines)
                self.dbs[idx].append((state, reward, self.done, action))

                # Render
                if not self.use_gui:
                    self.stdscr.addstr(str(engine))
                    self.stdscr.addstr(f'reward: {self.engine_states[idx]}\n')

                if self.engine_states[idx]['KO'] >= self.KO_num_to_win:
                    game_over = True
            if self.use_gui:
                self.gui.update_screen()
            else:
                self.stdscr.addstr(f'Time: {time.time() - self.start_time:.1f}\n')
        self.compare_score()
        print(f"Winner: {self.winner} States: {self.engine_states}")

        return self.dbs

    def set_engine_state(self, idx, engine, reward, cleared_lines):
        self.engine_states[idx]['garbage_lines'] = engine.garbage_lines
        self.engine_states[idx]['highest_line'] = engine.highest_line
        self.engine_states[idx]['hold_locked'] = engine.hold_locked
        self.engine_states[idx]['hold_shape'] = engine.hold_shape
        self.engine_states[idx]['hold_shape_name'] = engine.hold_shape_name
        self.engine_states[idx]['shape_name'] = engine.shape_name
        self.engine_states[idx]['next_shape_name'] = engine.next_shape_name
        self.engine_states[idx]['reward'] += reward
        self.engine_states[idx]['lines_sent'] += cleared_lines


if __name__ == '__main__':
    args = parse_args()
    global_engine = GlobalEngine(args.width, args.height, args.player_num, args.use_gui, args.block_size)
    dbs = global_engine.play_game(args.step_to_final)