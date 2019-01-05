import argparse
import curses
import time
import signal
import sys

from engine import TetrisEngine
from gui.gui import GUI
from fixed_policy_agent import FixedPolicyAgent
from genetic_policy_agent import GeneticPolicyAgent
from random_agent import RandomActionAgent
from ac_agent import setup_model
from logging_config import logger

gen_agent = GeneticPolicyAgent()
random_agent = RandomActionAgent()
fixed_policy_agent = FixedPolicyAgent()
ac_agent = setup_model()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ww', '--width', type=int, default=10, help='Window width')
    parser.add_argument('-hh', '--height', type=int, default=16, help='Window height')
    parser.add_argument('-n', '--game_num', type=int, default=10, help='Number of games')
    parser.add_argument('-kn', '--KO_num_to_win', type=int, default=5, help='Number of KO to win a game')
    parser.add_argument('-p', '--players', nargs='+', default=['g', 'f'], help='List of player type')
    parser.add_argument('-b', '--block_size', type=int, default=30, help='Set block size to enlarge GUI')
    parser.add_argument('-g', '--use_gui', type=int, default=0, help='Active output to gui')
    args = parser.parse_args()
    return args


class GlobalEngine:
    def __init__(
        self, width, height, use_gui, players,
        block_size,
        KO_num_to_win, game_num, game_time=120,
    ):
        self.width = width
        self.height = height
        self.player_num = len(players)
        self.game_time = game_time
        self.KO_num_to_win = KO_num_to_win
        self.game_num = game_num

        # For GUI use
        self.gui = None
        self.block_size = block_size
        self.use_gui = use_gui
        self.pause = False

        self.use_keyboard = 'k' in players
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
        self.stats = {}
        self.players = {}
        for i in range(len(players)):
            if players[i] == 'k':
                self.players[i] = self
            elif players[i] == 'f':
                self.players[i] = fixed_policy_agent
            elif players[i] == 'r':
                self.players[i] = random_agent
            elif players[i] == 'g':
                self.players[i] = gen_agent
            elif players[i] == 'a':
                self.players[i] = ac_agent
            else:
                raise NotImplementedError(f"{players}")
            self.stats[i] = {}
            self.stats[i]['win_times'] = 0
            self.stats[i]['total_sent_line'] = 0
            self.stats[i]['total_cleared_line'] = 0
            self.stats[i]['total_KO'] = 0

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

    def get_action(self, engine, shape, anchor, board):
        return self.get_action_from_keyboard()

    def get_player_action(self, engine_idx):
        engine = self.engines[engine_idx]
        try:
            return self.players[engine_idx].get_action(
                engine, engine.shape, engine.anchor, engine.board
            )
        except Exception as err:
            logger.error(err)
            import pdb
            pdb.set_trace()
            logger.error(err)

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

    def play(self):
        for i in range(self.game_num):
            self.play_game()

    def play_game(self):
        # Initialization
        self.setup()

        game_over = False
        while time.time() - self.start_time < self.game_time and not game_over:
            self.update_screen()
            game_over = self.update_engines()

        winner, max_score = self.compare_score()
        self.update_stats(winner)
        if not self.use_gui:
            self.stdscr.clear()
            self.stdscr.addstr(f'Game Over, winner: {winner}, States: {self.engine_states}\n')
        else:
            logger.info(f"Winner: {winner}")
            logger.info(f"States: {self.engine_states}")

        return winner

    def update_stats(self, winner):
        self.stats[winner]['win_times'] += 1
        for i in range(self.player_num):
            self.stats[i]['total_sent_line'] += self.engine_states[i]['lines_sent']
            self.stats[i]['total_cleared_line'] += self.engine_states[i]['lines_cleared']
            self.stats[i]['total_KO'] += self.engine_states[i]['KO']

    def get_stats(self):
        for i in range(self.player_num):
            self.stats[i]['avg_sent_line'] = self.stats[i]['total_sent_line'] / self.game_num
            self.stats[i]['avg_cleared_line'] = self.stats[i]['total_cleared_line'] / self.game_num
        return self.stats

    def show_stats(self):
        stats = self.get_stats()
        stats_string = ""
        for i in range(self.player_num):
            stats_string += f"Player {i+1}: Win {stats[i]['win_times']}\n"
            stats_string += f"\tAvg line sent: {stats[i]['avg_sent_line']}\n"
            stats_string += f"\tAvg line cleared: {stats[i]['avg_cleared_line']}\n"
            stats_string += f"\tTotal line sent: {stats[i]['total_sent_line']}\n"
            stats_string += f"\tTotal line cleared: {stats[i]['total_cleared_line']}\n"
            stats_string += f"\tTotal KO: {stats[i]['total_KO']}\n\n"
        return stats_string

    def update_engines(self):
        game_over = False
        for idx, engine in self.engines.items():
            action = self.get_player_action(idx)

            # Game step
            state, reward, self.done, cleared_lines, sent_lines = engine.step(action)

            # Update state
            self.set_engine_state(idx, engine, reward)
            self.send_lines(idx, sent_lines)
            # self.dbs[idx].append((state, reward, self.done, action))

            if self.engine_states[idx]['KO'] >= self.KO_num_to_win:
                game_over = True
        return game_over

    def update_screen(self):
        if self.use_gui:
            self.gui.update_screen()
        else:
            self.stdscr.clear()
            self.stdscr.addstr(f"Game {self.game_count}, Stat: {self.stats}\n")
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
        args.width, args.height,
        args.use_gui, args.players,
        args.block_size,
        args.KO_num_to_win, args.game_num
    )
    signal.signal(signal.SIGINT, global_engine.tear_down)
    global_engine.play()
    if not args.use_gui:
        global_engine.stdscr.clear()
        global_engine.stdscr.addstr(f"\n------------------------------------------------\n")
        global_engine.stdscr.addstr(f"{global_engine.game_count} games done\n\n")
        global_engine.stdscr.addstr(f"{global_engine.show_stats()}\n")
    else:
        logger.info(f"{global_engine.game_count} games done")
        logger.info(f"{global_engine.show_stats()}")
    global_engine.get_action_from_keyboard()
    global_engine.tear_down(None, None)
