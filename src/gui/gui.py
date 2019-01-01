# Based on:
# PYTRIS™ Copyright (c) 2017 Jason Kim All Rights Reserved.

import os

import pygame

from gui.mino import Tetrimino

running_gui = None
DIR_PATH = os.path.dirname(os.path.realpath(__file__))

pygame.init()


class UIVariables:
    framerate = 50  # Bigger -> Slower
    shape_names = ['T', 'J', 'L', 'Z', 'S', 'I', 'O']

    # Fonts
    font_path = os.path.join(DIR_PATH, "fonts/OpenSans-Light.ttf")
    font_path_b = os.path.join(DIR_PATH, "fonts/OpenSans-Bold.ttf")
    font_path_i = os.path.join(DIR_PATH, "fonts/Inconsolata/Inconsolata.otf")

    h1 = pygame.font.Font(font_path, 50)
    h2 = pygame.font.Font(font_path, 30)
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
    grey_1 = (26, 26, 26)    # rgb(26, 26, 26)
    grey_2 = (35, 35, 35)    # rgb(35, 35, 35)
    grey_3 = (55, 55, 55)    # rgb(55, 55, 55)
    grey_4 = (75, 75, 75)

    # Tetrimino colors
    cyan = (69, 206, 204)    # rgb(69, 206, 204) # I
    blue = (64, 111, 249)    # rgb(64, 111, 249) # J
    orange = (253, 189, 53)  # rgb(253, 189, 53) # L
    yellow = (246, 227, 90)  # rgb(246, 227, 90) # O
    green = (98, 190, 68)    # rgb(98, 190, 68) # S
    pink = (242, 64, 235)    # rgb(242, 64, 235) # T
    red = (225, 13, 27)      # rgb(225, 13, 27) # Z

    t_color = {
        -2: grey_4,
        -1: grey_3,
        0: grey_2,
        1: cyan,
        2: blue,
        3: orange,
        4: yellow,
        5: green,
        6: pink,
        7: red,
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
        pygame.draw.rect(
            self.screen,
            UIVariables.grey_1,
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
        text_player = UIVariables.h2.render("Player "+str(player_id+1), 1, UIVariables.black)
        text_hold = UIVariables.h5.render("HOLD", 1, UIVariables.black)
        text_next = UIVariables.h5.render("NEXT", 1, UIVariables.black)
        text_combo = UIVariables.h5.render("Combo", 1, UIVariables.black)
        combo_value = UIVariables.h4.render(str(self.global_state.engine_states[player_id]['combo']),
                                            1, UIVariables.black)
        text_lines_sent = UIVariables.h5.render("Lines sent", 1, UIVariables.black)
        lines_sent_value = UIVariables.h4.render(str(self.global_state.engine_states[player_id]['lines_sent']),
                                                 1, UIVariables.black)
        text_lines_cleared = UIVariables.h5.render("Lines cleared", 1, UIVariables.black)
        lines_cleared_value = UIVariables.h4.render(str(self.global_state.engine_states[player_id]['lines_cleared']),
                                                    1, UIVariables.black)
        text_ko = UIVariables.h5.render("KO's", 1, UIVariables.black)
        ko_value = UIVariables.h4.render(str(self.global_state.engines[player_id].n_deaths),
                                         1, UIVariables.black)

        # Place texts
        self.screen.blit(text_player, (x_start + 4 * self.block_size + 120, y_start + 10))
        self.screen.blit(text_hold, (x_start + 15, y_start + 60))
        self.screen.blit(text_next, (x_start + 15, y_start + 60 + 5 * self.block_size))
        self.screen.blit(text_combo, (x_start + 15, y_start + 60 + 10 * self.block_size))
        self.screen.blit(combo_value, (x_start + 20, y_start + 60 + 10 * self.block_size + 16))
        self.screen.blit(text_lines_sent, (x_start + 15, y_start + 60 + 10 * self.block_size + 60))
        self.screen.blit(lines_sent_value, (x_start + 20, y_start + 60 + 10 * self.block_size + 76))
        self.screen.blit(text_lines_cleared, (x_start + 15, y_start + 60 + 10 * self.block_size + 120))
        self.screen.blit(lines_cleared_value, (x_start + 20, y_start + 60 + 10 * self.block_size + 136))
        self.screen.blit(text_ko, (x_start + 15, y_start + 60 + 10 * self.block_size + 180))
        self.screen.blit(ko_value, (x_start + 20, y_start + 60 + 10 * self.block_size + 196))

    # Draw next mino for one player
    def _draw_next_mino(self, x_start, y_start, player_id):
        next_mino_name = self.global_state.engines[player_id].next_shape_name
        next_mino = UIVariables.shape_names.index(next_mino_name)
        grid_n = Tetrimino.mino_map[next_mino][0]
        for i in range(4):
            for j in range(4):
                dx = x_start + 20 + self.block_size * j
                dy = y_start + 60 + 5 * self.block_size + self.block_size + self.block_size * i
                if grid_n[i][j] != 0:
                    pygame.draw.rect(
                        self.screen,
                        UIVariables.t_color[grid_n[i][j]],
                        pygame.Rect(dx, dy, self.block_size, self.block_size)
                    )

    # Draw hold mino for one player
    def _draw_hold_mino(self, x_start, y_start, player_id):
        hold_mino_name = self.global_state.engines[player_id].hold_shape_name
        if hold_mino_name is not None:
            hold_mino = UIVariables.shape_names.index(hold_mino_name)
            grid_h = Tetrimino.mino_map[hold_mino][0]
            for i in range(4):
                for j in range(4):
                    dx = x_start + 20 + self.block_size * j
                    dy = y_start + 60 + self.block_size + self.block_size * i
                    if grid_h[i][j] != 0:
                        pygame.draw.rect(
                            self.screen,
                            UIVariables.t_color[grid_h[i][j]],
                            pygame.Rect(dx, dy, self.block_size, self.block_size)
                        )

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
        self.screen.fill(UIVariables.grey_1)

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
                    key_pressed = event.key
                    return key_pressed

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
