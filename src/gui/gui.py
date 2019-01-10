# Based on:
# PYTRIS™ Copyright (c) 2017 Jason Kim All Rights Reserved.

import os

import pygame

from gui.mino import Tetrimino

running_gui = None
DIR_PATH = os.path.dirname(os.path.realpath(__file__))

pygame.init()


def get_brighter_color(color, factor=1.2):
    brighter_color = tuple([int(x * factor) if int(x * factor) < 255 else 255 for x in color])
    return brighter_color


class UIVariables:
    framerate = 50  # Bigger -> Slower
    shape_names = ['T', 'J', 'L', 'Z', 'S', 'I', 'O']

    # Fonts
    font_path = os.path.join(DIR_PATH, "fonts/OpenSans-Light.ttf")
    font_path_b = os.path.join(DIR_PATH, "fonts/OpenSans-Bold.ttf")
    font_path_i = os.path.join(DIR_PATH, "fonts/Inconsolata/Inconsolata.otf")

    h1 = pygame.font.Font(font_path, 50)
    h2 = pygame.font.Font(font_path, 30)
    h3 = pygame.font.Font(font_path, 25)
    h4 = pygame.font.Font(font_path, 20)
    h5 = pygame.font.Font(font_path, 13)
    h6 = pygame.font.Font(font_path, 10)

    h1_b = pygame.font.Font(font_path_b, 50)
    h2_b = pygame.font.Font(font_path_b, 30)

    h2_i = pygame.font.Font(font_path_i, 30)
    h5_i = pygame.font.Font(font_path_i, 13)

    # Background colors
    black = (10, 10, 10)     # rgb(10, 10, 10)
    white = (255, 255, 255)  # rgb(255, 255, 255)
    background = (200, 255, 200)
    grey_garbage = (119, 120, 115)
    grey_game_background = (75, 75, 75)
    grey_boarder = (26, 26, 26)
    grey_board = (35, 35, 35)

    # Tetrimino colors
    cyan = (11, 160, 228)    # I
    blue = (33, 68, 198)     # J
    orange = (216, 93, 13)   # L
    yellow = (224, 154, 0)   # O
    green = (89, 175, 16)    # S
    pink = (185, 32, 138)    # T
    red = (200, 15, 46)      # Z

    t_color = {
        -2: grey_game_background,
        -1: grey_garbage,
        0: grey_board,
        1: pink,
        2: blue,
        3: orange,
        4: red,
        5: green,
        6: cyan,
        7: yellow,
    }


class GUI:
    # Get global engine and setup gui
    def __init__(self, global_state, block_size):
        self.global_state = global_state

        # Initial values
        self.blink = False
        self.pause = False
        self.start = True
        self.game_over = False
        self.done = False
        self.pressed_key = None

        self.block_size = block_size

        self.screen_width = global_state.player_num * (global_state.width * block_size +
                                                       4 * block_size + 80)
        self.screen_height = 80 + global_state.height * block_size
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.time.set_timer(pygame.USEREVENT, UIVariables.framerate * 10)
        pygame.display.set_caption("Make Tetris Great Again")

    # Draw block
    def _draw_block(self, x, y, color):
        pygame.draw.rect(
            self.screen,
            color,
            pygame.Rect(x, y, self.block_size, self.block_size)
        )
        inner_pad = self.block_size // 7
        pygame.draw.rect(
            self.screen,
            get_brighter_color(color),
            pygame.Rect(x+inner_pad, y+inner_pad, self.block_size-inner_pad*2, self.block_size-inner_pad*2),
            inner_pad
        )
        r, g, b = color
        if not (r == g and g == b):
            pygame.draw.rect(
                self.screen,
                get_brighter_color(color, 2),
                pygame.Rect(x+1, y+1, 3, 3),
                3
            )
        pygame.draw.rect(
            self.screen,
            UIVariables.grey_boarder,
            pygame.Rect(x, y, self.block_size, self.block_size),
            1
        )

    # Draw board of one player
    def _draw_board(self, x_start, y_start, player_id):
        board = self.global_state.engines[player_id].get_board()
        for x in range(self.global_state.width):
            for y in range(self.global_state.height):
                dx = x_start + 4 * self.block_size + 40 + self.block_size * x
                dy = y_start + 60 + self.block_size * y
                self._draw_block(dx, dy, UIVariables.t_color[int(board[x][y])])

    # Draw score bar for one player
    def _draw_score(self, x_start, y_start, player_id):
        # Draw sidebar
        pygame.draw.rect(
            self.screen,
            UIVariables.white,
            pygame.Rect(x_start, y_start, x_start + self.screen_width / self.global_state.player_num,
                        y_start + self.screen_height)
        )

        # Draw texts
        player_type = (
            'Keyboard'
            if 'Agent' not in self.global_state.players[player_id].__class__.__name__
            else self.global_state.players[player_id].__class__.__name__
        )
        text_player = UIVariables.h3.render(
            f"P{player_id+1}: {player_type}", 1, UIVariables.black)
        text_hold = UIVariables.h5.render("HOLD", 1, UIVariables.black)
        text_next = UIVariables.h5.render("NEXT", 1, UIVariables.black)

        # Place texts
        pad = 35
        self.screen.blit(text_player, (x_start + 5 * self.block_size + 10, y_start + 20))
        self.screen.blit(text_hold, (x_start + 15, y_start + pad + 5))
        self.screen.blit(text_next, (x_start + 15, y_start + pad + 5 + 5 * self.block_size))
        texts = [
            "K.O.",
            f"  {self.global_state.engine_states[player_id]['KO']}",
            "Combo",
            f"  {self.global_state.engine_states[player_id]['combo']}",
            "Lines sent",
            f"  {self.global_state.engine_states[player_id]['lines_sent']}",
            "Lines cleared",
            f"  {self.global_state.engine_states[player_id]['lines_cleared']}"
        ]
        for i, text in enumerate(texts):
            if i % 2 == 0:
                text_render = UIVariables.h5.render(text, 1, UIVariables.black)
            else:
                text_render = UIVariables.h4.render(text, 1, UIVariables.black)
            self.screen.blit(
                text_render, (x_start + 15, y_start + 20 + 10 * self.block_size + (i+1) * pad - (i % 2) * 20))

    # Draw next mino for one player
    def _draw_next_mino(self, x_start, y_start, player_id):
        pygame.draw.rect(
            self.screen,
            UIVariables.black,
            pygame.Rect(x_start + 20, y_start + 65 + 5 * self.block_size, self.block_size * 4, self.block_size * 4),
        )
        next_mino_name = self.global_state.engines[player_id].next_shape_name
        next_mino = UIVariables.shape_names.index(next_mino_name)
        grid_n = Tetrimino.mino_map[next_mino][0]
        for i in range(4):
            for j in range(4):
                dx = x_start + 20 + self.block_size * j
                dy = y_start + 60 + 5 * self.block_size + self.block_size + self.block_size * i
                if grid_n[i][j] != 0:
                    self._draw_block(dx, dy, UIVariables.t_color[grid_n[i][j]])

    # Draw hold mino for one player
    def _draw_hold_mino(self, x_start, y_start, player_id):
        pygame.draw.rect(
            self.screen,
            UIVariables.black,
            pygame.Rect(x_start + 20, y_start + 65, self.block_size * 4, self.block_size * 4),
        )
        hold_mino_name = self.global_state.engines[player_id].hold_shape_name
        if hold_mino_name is not None:
            hold_mino = UIVariables.shape_names.index(hold_mino_name)
            grid_h = Tetrimino.mino_map[hold_mino][0]
            for i in range(4):
                for j in range(4):
                    dx = x_start + 20 + self.block_size * j
                    dy = y_start + 60 + self.block_size + self.block_size * i
                    if grid_h[i][j] != 0:
                        self._draw_block(dx, dy, UIVariables.t_color[grid_h[i][j]])

    # Draw game screen of one player
    def _draw_one_screen(self, x_start, y_start, player_id):
        # Draw score bar
        self._draw_score(x_start, y_start, player_id)

        # Draw next mino
        self._draw_next_mino(x_start, y_start, player_id)

        # Draw hold mino
        self._draw_hold_mino(x_start, y_start, player_id)

        # Draw board
        self._draw_board(x_start, y_start, player_id)

    # Render all players screen
    def _draw_all_screen(self):
        # Background gray
        self.screen.fill(UIVariables.black)

        if self.global_state.player_num > 4:
            print("Max player number 4")
            pygame.quit()
            exit(10)

        for t in range(0, self.global_state.player_num):
            self._draw_one_screen(t * self.screen_width / self.global_state.player_num, 0, t)
        pygame.display.flip()

    # TODO
    def _draw_statistics(self):
        return

    # Updates the whole screen on call
    def update_screen(self):
        pygame.time.wait(UIVariables.framerate)
        self._draw_all_screen()

    # Get last keyboard input since last calling this function
    def last_gui_input(self):
        # Pause screen
        if self.pause:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.done = True
                elif event.type == pygame.USEREVENT:
                    pygame.time.set_timer(pygame.USEREVENT, 300)
                    self._draw_all_screen()

                    pause_text = UIVariables.h2_b.render("PAUSED", 1, UIVariables.white)
                    pause_start = UIVariables.h5.render("Press esc to continue", 1, UIVariables.white)

                    self.screen.blit(pause_text, (43, 100))
                    if self.blink:
                        self.screen.blit(pause_start, (40, 160))
                        self.blink = False
                    else:
                        self.blink = True
                    pygame.display.update()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.pause = False
                        self.global_state.pause = False
                        pygame.time.set_timer(pygame.USEREVENT, 1)

        # Game screen
        elif self.start:
            # Clear event stack and get the last one
            events = pygame.event.get()
            for event in reversed(events):
                if event.type == pygame.KEYDOWN:
                    self.pressed_key = event.key
                elif event.type == pygame.KEYUP:
                    self.pressed_key = None
                return self.pressed_key

        # Game over screen
        elif self.game_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.done = True
                elif event.type == pygame.USEREVENT:
                    pygame.time.set_timer(pygame.USEREVENT, 300)

                    self._draw_statistics()

                    # Show "Game over"
                    over_text_1 = UIVariables.h2_b.render("GAME", 1, UIVariables.white)
                    over_text_2 = UIVariables.h2_b.render("OVER", 1, UIVariables.white)
                    self.screen.blit(over_text_1, (58, 75))
                    self.screen.blit(over_text_2, (62, 105))

                    over_start = UIVariables.h5.render("Press return to exit", 1, UIVariables.white)
                    if self.blink:
                        self.screen.blit(over_start, (32, 195))
                        self.blink = False
                    else:
                        self.blink = True
                    pygame.display.update()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
