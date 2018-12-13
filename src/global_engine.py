import curses
import time

from engine import TetrisEngine


class GlobalEngine:
    def __init__(
        self, width, height, player_num, game_time=120,
        KO_num_to_win=2
    ):
        self.width = width
        self.height = height
        self.player_num = player_num
        self.game_time = game_time
        self.KO_num_to_win = KO_num_to_win
        self.winner = None

        self.engines = {}
        for i in range(player_num):
            self.engines[i] = TetrisEngine(width, height)
            self.engines[i].clear()

        self.global_state = {}
        self.engine_states = {}

    def keyboard_control(self, key):
        if key == ord('a'):  # Shift left
            action = 0
        elif key == ord('d'):  # Shift right
            action = 1
        elif key == ord('w'):  # Hard drop
            action = 2
        elif key == ord('s'):  # Soft drop
            action = 3
        elif key == ord('q'):  # Rotate left
            action = 4
        elif key == ord('e'):  # Rotate right
            action = 5
        elif key == ord('f'):  # Hold
            action = 7
        else:
            action = 6
        return action

    def sent_lines(self, idx, cleared_lines):
        for other_idx, other_engine in self.engines.items():
            if other_idx != idx:
                other_engine.receive_bomb_lines(cleared_lines)

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

    def play_game(self):
        stdscr = curses.initscr()

        # Store play information
        dbs = {}

        done = False

        # Initialization
        for i in range(self.player_num):
            # Initial rendering
            stdscr.addstr(str(self.engines[i]))
            self.engine_states[i] = {
                "KO": 0,
                "reward": 0,
                "lines_sent": 0,
                "hold_shape": None,
                "hold_shape_name": None,
                "holded": False,
                "bomb_lines": 0,
                "highest_line": 0
            }
            # Initialize dbs
            dbs[i] = []
            # Global action

        self.start_time = time.time()
        game_over = False
        while time.time() - self.start_time < self.game_time and not game_over:
            key = stdscr.getch()
            action = self.keyboard_control(key)

            stdscr.clear()
            for idx, engine in self.engines.items():
                # Game step
                state, reward, done, cleared_lines = engine.step(action)
                self.engine_states[idx]['lines_sent'] += cleared_lines
                self.engine_states[idx]['bomb_lines'] = engine.bomb_lines
                self.engine_states[idx]['highest_line'] = engine.highest_line
                self.engine_states[idx]['holded'] = engine.holded
                self.engine_states[idx]['hold_shape'] = engine.hold_shape
                self.engine_states[idx]['hold_shape_name'] = engine.hold_shape_name
                self.engine_states[idx]['shape_name'] = engine.shape_name
                self.engine_states[idx]['next_shape_name'] = engine.next_shape_name
                self.sent_lines(idx, cleared_lines)
                self.engine_states[idx]['reward'] += reward
                dbs[idx].append((state, reward, done, action))

                # Render
                stdscr.addstr(str(engine))
                stdscr.addstr(f'reward: {self.engine_states[idx]}\n')

                if self.engine_states[idx]['KO'] >= self.KO_num_to_win:
                    game_over = True
            stdscr.addstr(f'Time: {time.time() - self.start_time:.1f}\n')
        self.compare_score()
        print(f"Winner: {self.winner} States: {self.engine_states}")

        return dbs


if __name__ == '__main__':
    width, height = 10, 16  # standard tetris friends rules
    player_num = 2
    global_engine = GlobalEngine(width, height, player_num)
    dbs = global_engine.play_game()
    # print(dbs)
