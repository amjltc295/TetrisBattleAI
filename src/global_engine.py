import argparse
import curses
import time
import signal
import sys

from engine import TetrisEngine
from gui.gui import GUI
import fixed_policy_agent
from genetic_policy_agent import GeneticPolicyAgent
from logging_config import logger

gen_agent = GeneticPolicyAgent()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ww', '--width', type=int, default=10, help='Window width')
    parser.add_argument('-hh', '--height', type=int, default=16, help='Window height')
    parser.add_argument('-p', '--player_num', type=int, default=2, help='Number of players')
    parser.add_argument('-n', '--game_num', type=int, default=10, help='Number of games')
    parser.add_argument('-kn', '--KO_num_to_win', type=int, default=5, help='Number of KO to win a game')
    parser.add_argument('-k', '--use_keyboard', default=False, action='store_true',
                        help='Use keyboard input (if not, use GA instead)')
    parser.add_argument('-b', '--block_size', type=int, default=30, help='Set block size to enlarge GUI')
    parser.add_argument('-g', '--use_gui', type=int, default=0, help='Active output to gui')
    args = parser.parse_args()
    return args


class GlobalEngine:
    def __init__(
        self, width, height, player_num, use_gui, use_keyboard,
        block_size,
        KO_num_to_win, game_num, game_time=120,
    ):
        self.width = width
        self.height = height
        self.player_num = player_num
        self.game_time = game_time
        self.KO_num_to_win = KO_num_to_win

        # For GUI use
        self.gui = None
        self.block_size = block_size
        self.use_gui = use_gui
        self.pause = False

        self.use_keyboard = use_keyboard

        self.key_action_map = {
            ord('a'): "move_left",  # Shift left
            ord('d'): "move_right",  # Shift right
            ord('w'): "hard_drop",  # Hard drop
            ord('s'): "soft_drop",  # Soft drop
            ord('q'): "rotate_left",  # Rotate left
            ord('e'): "rotate_right",  # Rotate right
            ord('f'): "hold"   # Hold
        }
        self.engines = {}
        self.win_times = {}
        self.players = {}
        for i in range(player_num):
            if i == 0:
                if use_keyboard:
                    self.players[i] = 'keyboard'
                else:
                    self.players[i] = 'genetic_policy_agent'
            else:
                self.players[i] = 'fixed_policy_agent'
            self.win_times[i] = 0

        self.engine_states = {}
        self.game_count = 0

    def setup(self):
        # Initialization
        for i in range(self.player_num):
            self.engines[i] = TetrisEngine(self.width, self.height)
            self.engines[i].clear()
        if self.use_gui:
            gui = GUI(self, self.block_size)
            self.gui = gui
        else:
            self.stdscr = curses.initscr()
            curses.noecho()

        # Store play information
        self.dbs = {}

        self.done = False

        for i in range(self.player_num):
            # Initial rendering
            self.engine_states[i] = {
                "KO": 0,
                "reward": 0,
                "lines_sent": 0,
                "lines_cleared": 0,
                "hold_shape": None,
                "hold_shape_name": None,
                "hold_locked": False,
                "garbage_lines": 0,
                "highest_line": 0,
                "combo": -1
            }
            # Initialize dbs
            self.dbs[i] = []

        self.game_count += 1
        self.start_time = time.time()

    def keyboard_control(self, key):
        if key in self.key_action_map:
            action = self.key_action_map[key]
        else:
            action = 'idle'
        return action

    def send_lines(self, idx, lines_to_send):
        for other_idx, other_engine in self.engines.items():
            if other_idx != idx:
                other_engine.receive_garbage_lines(lines_to_send)

                # Get KO
                if self.player_num == 2:
                    self.engine_states[idx]['KO'] = other_engine.n_deaths
                elif lines_to_send > 0:
                    if not other_engine.is_alive():
                        self.engine_states[idx]['KO'] += 1

    def compare_score(self):
        max_score = 0
        for idx, engine in self.engines.items():
            score = self.engine_states[idx]['KO'] * 10000 + self.engine_states[idx]['lines_sent']
            if score > max_score:
                winner = idx
                max_score = score
        return winner, max_score

    def get_action(self, engine_idx):
        playert_type = self.players[engine_idx]
        if playert_type == 'keyboard':
            return self.get_action_from_keyboard()
        elif playert_type == 'fixed_policy_agent':
            return self.get_action_from_fixed_policy_agent(self.engines[engine_idx])
        elif playert_type == 'genetic_policy_agent':
            return self.get_action_from_genetic_policy_agent(self.engines[engine_idx])
        else:
            raise NotImplementedError(f"Player type {playert_type} not exists")

    def get_action_from_fixed_policy_agent(self, engine):
        action = fixed_policy_agent.agent.get_action(
            engine, engine.shape, engine.anchor, engine.board
        )
        return action

    def get_action_from_genetic_policy_agent(self, engine):
        action = gen_agent.get_action(
            engine, engine.shape, engine.anchor, engine.board
        )
        return action

    def get_action_from_keyboard(self):
        def get_key():
            if self.use_gui:
                key = self.gui.last_gui_input()
            else:
                key = self.stdscr.getch()
            return key
        key = get_key()
        action = self.keyboard_control(key)
        return action

    def play_game(self):
        # Initialization
        self.setup()

        game_over = False
        while time.time() - self.start_time < self.game_time and not game_over:
            self.update_screen()
            game_over = self.update_engines()

        winner, max_score = self.compare_score()
        self.win_times[winner] += 1
        if not self.use_gui:
            self.stdscr.clear()
            self.stdscr.addstr(f'Game Over, winner: {winner}, States: {self.engine_states}\n')
        else:
            logger.info(f"Winner: {winner}")
            logger.info(f"States: {self.engine_states}")

        return self.dbs, winner

    def update_engines(self):
        game_over = False
        for idx, engine in self.engines.items():
            action = self.get_action(idx)

            # Game step
            state, reward, self.done, cleared_lines, sent_lines = engine.step(action)

            # Update state
            self.set_engine_state(idx, engine, reward)
            self.send_lines(idx, sent_lines)
            self.dbs[idx].append((state, reward, self.done, action))

            if self.engine_states[idx]['KO'] >= self.KO_num_to_win:
                game_over = True
        return game_over

    def update_screen(self):
        if self.use_gui:
            self.gui.update_screen()
        else:
            self.stdscr.clear()
            self.stdscr.addstr(f"Game {self.game_count}, Win times: {self.win_times}\n")
            for idx, engine in self.engines.items():
                self.stdscr.addstr(str(engine))
                self.stdscr.addstr(f'reward: {self.engine_states[idx]}\n')
            self.stdscr.addstr(f'Time: {time.time() - self.start_time:.1f}\n')
            self.stdscr.refresh()
        if self.use_keyboard:
            time.sleep(0.05)

    def set_engine_state(self, idx, engine, reward):
        self.engine_states[idx]['combo'] = engine.combo
        self.engine_states[idx]['lines_sent'] = engine.total_sent_lines
        self.engine_states[idx]['garbage_lines'] = engine.garbage_lines
        self.engine_states[idx]['highest_line'] = engine.highest_line
        self.engine_states[idx]['hold_locked'] = engine.hold_locked
        self.engine_states[idx]['hold_shape'] = engine.hold_shape
        self.engine_states[idx]['hold_shape_name'] = engine.hold_shape_name
        self.engine_states[idx]['shape_name'] = engine.shape_name
        self.engine_states[idx]['next_shape_name'] = engine.next_shape_name
        self.engine_states[idx]['reward'] += reward
        self.engine_states[idx]['lines_cleared'] = engine.total_cleared_lines

    def tear_down(self, sig, frame):
        if not self.use_gui:
            curses.endwin()
        sys.exit(0)


if __name__ == '__main__':
    args = parse_args()
    global_engine = GlobalEngine(
        args.width, args.height, args.player_num,
        args.use_gui, args.use_keyboard,
        args.block_size,
        args.KO_num_to_win, args.game_num
    )
    signal.signal(signal.SIGINT, global_engine.tear_down)
    for i in range(args.game_num):
        dbs, winner = global_engine.play_game()
    if not args.use_gui:
        global_engine.stdscr.clear()
        global_engine.stdscr.addstr(f"\n------------------------------------------------\n")
        global_engine.stdscr.addstr(f"{global_engine.game_count} games done, Win times: {global_engine.win_times}\n")
    else:
        logger.info(f"{global_engine.game_count} games done, Win times: {global_engine.win_times}\n")
    global_engine.get_action_from_keyboard()
    global_engine.tear_down(None, None)
