import curses

from engine import TetrisEngine


class GlobalEngine:
    def __init__(
        self, width, height, player_num
    ):
        self.width = width
        self.height = height
        self.player_num = player_num

        self.engines = {}
        for i in range(player_num):
            self.engines[i] = TetrisEngine(width, height)
            self.engines[i].clear()

    def play_game(self):
        stdscr = curses.initscr()

        # Store play information
        dbs = {}

        done = False

        player_actions = {}
        for i in range(self.player_num):
            # Initial rendering
            stdscr.addstr(str(self.engines[i]))
            # Initialize dbs
            dbs[i] = []
            # Global action
            player_actions[i] = 6

        while not done:
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
                state, reward, done = engine.step(action)
                dbs[idx].append((state, reward, done, action))

                # Render
                stdscr.addstr(str(engine))
                stdscr.addstr(f'reward: reward\n')

        return dbs


if __name__ == '__main__':
    width, height = 10, 20  # standard tetris friends rules
    player_num = 2
    global_engine = GlobalEngine(width, height, player_num)
    dbs = global_engine.play_game()
    print(dbs)
