import curses

from engine import TetrisEngine


INIT_ENGINE_STATE = {
    "KO": 0,
    "reward": 0,
    "lines_sent": 0,
    "hold_block": None,
    "holded": False,
    "bomb_lines": 0
}


class GlobalEngine:
    def __init__(
        self, width, height, player_num, count_down=120
    ):
        self.width = width
        self.height = height
        self.player_num = player_num
        self.count_down = 120

        self.engines = {}
        for i in range(player_num):
            self.engines[i] = TetrisEngine(width, height)
            self.engines[i].clear()

        self.global_state = {}
        self.engine_states = {}

    def play_game(self):
        stdscr = curses.initscr()

        # Store play information
        dbs = {}

        done = False

        player_actions = {}
        # Initialization
        for i in range(self.player_num):
            # Initial rendering
            stdscr.addstr(str(self.engines[i]))
            self.engine_states[i] = {
                "KO": 0,
                "reward": 0,
                "lines_sent": 0,
                "hold_block": None,
                "holded": False,
                "bomb_lines": 0
            }
            # Initialize dbs
            dbs[i] = []
            # Global action
            player_actions[i] = 6

        while self.count_down > 0:
            self.count_down -= 1
            action = 6
            key = stdscr.getch()

            if key == -1:  # No key pressed
                action = 6
            elif key == ord('a'):  # Shift left
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

            stdscr.clear()
            for idx, engine in self.engines.items():
                # Game step
                state, reward, done, cleared_lines = engine.step(action)
                self.engine_states[idx]['lines_sent'] += cleared_lines
                if cleared_lines > 0:
                    for other_idx, other_engine in self.engines.items():
                        if other_idx != idx:
                            other_engine.receive_bomb_lines(cleared_lines)
                            self.engine_states[other_idx]['bomb_lines'] = other_engine.bomb_lines
                            if not other_engine.is_alive():
                                self.engine_states[idx]['KO'] += 1

                self.engine_states[idx]['reward'] += reward
                dbs[idx].append((state, reward, done, action))

                # Render
                stdscr.addstr(str(engine))
                stdscr.addstr(f'reward: {self.engine_states[idx]}\n')
            stdscr.addstr(f'Time: {self.count_down}\n')

        return dbs


if __name__ == '__main__':
    width, height = 10, 16  # standard tetris friends rules
    player_num = 2
    global_engine = GlobalEngine(width, height, player_num)
    dbs = global_engine.play_game()
    print(dbs)
