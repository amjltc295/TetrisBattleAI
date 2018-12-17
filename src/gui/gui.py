# Based on:
# PYTRIS™ Copyright (c) 2017 Jason Kim All Rights Reserved.

import pygame
from src.global_engine import GlobalEngine
from src.gui.mino import Tetrimino


pygame.init()


class UIVariables:
    block_size = 17  # Height, width of single block
    framerate = 30  # Bigger -> Slower

    # Fonts
    font_path = "./assets/fonts/OpenSans-Light.ttf"
    font_path_b = "./assets/fonts/OpenSans-Bold.ttf"
    font_path_i = "./assets/fonts/Inconsolata/Inconsolata.otf"

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

    # Tetrimino colors
    cyan = (69, 206, 204)    # rgb(69, 206, 204) # I
    blue = (64, 111, 249)    # rgb(64, 111, 249) # J
    orange = (253, 189, 53)  # rgb(253, 189, 53) # L
    yellow = (246, 227, 90)  # rgb(246, 227, 90) # O
    green = (98, 190, 68)    # rgb(98, 190, 68) # S
    pink = (242, 64, 235)    # rgb(242, 64, 235) # T
    red = (225, 13, 27)      # rgb(225, 13, 27) # Z

    t_color = [grey_2, cyan, blue, orange, yellow, green, pink, red, grey_3]


class GUI:
    # Get global engine and setup gui
    def __init__(self, global_state):
        self.global_state = global_state

        # Initial values
        self.blink = False
        self.pause = False
        self.start = True
        self.game_over = False
        self.done = False

        self.screen_width = 600
        self.screen_height = 400
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.time.set_timer(pygame.USEREVENT, UIVariables.framerate * 10)
        pygame.display.set_caption("Make Tetris Great Again")

    # Draw block
    def draw_block(self, x, y, color):
        pygame.draw.rect(
            self.screen,
            color,
            pygame.Rect(x, y, UIVariables.block_size, UIVariables.block_size)
        )
        pygame.draw.rect(
            self.screen,
            UIVariables.grey_1,
            pygame.Rect(x, y, UIVariables.block_size, UIVariables.block_size),
            1
        )

    def global_to_gui_number(self, number):
        if number <= 6:
            return number
        else:
            return 6

    # Draw board of one player
    def draw_board(self, x_start, y_start, player_id):
        board = self.global_state.engines[player_id].board
        for x in range(self.global_state.width):
            for y in range(self.global_state.height):
                dx = x_start + 96 + UIVariables.block_size * x
                dy = y_start + 0 + UIVariables.block_size * y
                self.draw_block(dx, dy, UIVariables.t_color[board[x][y + 1]])

    # Draw score bar for one player
    def draw_score(self, x_start, y_start, player_id):
        # Draw sidebar
        pygame.draw.rect(
            self.screen,
            UIVariables.white,
            pygame.Rect(x_start, y_start, x_start + 96, y_start + 400)
        )

        # Draw texts
        text_hold = UIVariables.h5.render("HOLD", 1, UIVariables.black)
        text_next = UIVariables.h5.render("NEXT", 1, UIVariables.black)
        text_score = UIVariables.h5.render("SCORE", 1, UIVariables.black)
        score_value = UIVariables.h4.render(str(self.global_state.engines[player_id].score),
                                            1, UIVariables.black)
        text_lines = UIVariables.h5.render("Total cleared lines", 1, UIVariables.black)
        lines_value = UIVariables.h4.render(str(self.global_state.engines[player_id].total_cleared_lines),
                                            1, UIVariables.black)
        text_ko = UIVariables.h5.render("KO's", 1, UIVariables.black)
        ko_value = UIVariables.h4.render(str(self.global_state.engines[player_id].n_deaths),
                                         1, UIVariables.black)

        # Place texts
        self.screen.blit(text_hold, (x_start + 15, y_start + 14))
        self.screen.blit(text_next, (x_start + 15, y_start + 104))
        self.screen.blit(text_score, (x_start + 15, y_start + 194))
        self.screen.blit(score_value, (x_start + 20, y_start + 210))
        self.screen.blit(text_lines, (x_start + 15, y_start + 254))
        self.screen.blit(lines_value, (x_start + 20, y_start + 270))
        self.screen.blit(text_ko, (x_start + 15, y_start + 314))
        self.screen.blit(ko_value, (x_start + 20, y_start + 330))

    # Draw next mino for one player
    def draw_next_mino(self, x_start, y_start, player_id):
        next_mino = self.global_state.engines[player_id].next_shape
        grid_n = Tetrimino.mino_map[next_mino][0]
        for i in range(4):
            for j in range(4):
                dx = x_start + 20 + UIVariables.block_size * j
                dy = y_start + 140 + UIVariables.block_size * i
                if grid_n[i][j] != 0:
                    pygame.draw.rect(
                        self.screen,
                        UIVariables.t_color[grid_n[i][j]],
                        pygame.Rect(dx, dy, UIVariables.block_size, UIVariables.block_size)
                    )

    # Draw hold mino for one player
    def draw_hold_mino(self, x_start, y_start, player_id):
        if self.global_state.engines[player_id].holded:
            hold_mino = self.global_state.engines[player_id].hold_shape
            grid_h = Tetrimino.mino_map[hold_mino][0]
            for i in range(4):
                for j in range(4):
                    dx = x_start + 20 + UIVariables.block_size * j
                    dy = y_start + 50 + UIVariables.block_size * i
                    if grid_h[i][j] != 0:
                        pygame.draw.rect(
                            self.screen,
                            UIVariables.t_color[grid_h[i][j]],
                            pygame.Rect(dx, dy, UIVariables.block_size, UIVariables.block_size)
                        )

    # Draw game screen of one player
    def draw_one_screen(self, x_start, y_start, player_id):
        # Background gray
        self.screen.fill(UIVariables.grey_1)

        # Draw score bar
        self.draw_score(x_start, y_start, player_id)

        # Draw next mino
        self.draw_next_mino(x_start, y_start, player_id)

        # Draw hold mino
        self.draw_hold_mino(x_start, y_start, player_id)

        # Draw board
        self.draw_board(x_start, y_start, player_id)

    # Render all players screen
    # Limited to 2 players right now
    def draw_all_screen(self):
        if self.global_state.player_num != 2:
            pygame.quit()
            exit(10)
        for t in range(0, self.global_state.player_num):
            self.draw_one_screen(t*400, 0, t)

    # TODO
    def draw_statistics(self):
                return

    def draw_whole_screen(self):
        # Pause screen
        if self.pause:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.done = True
                elif event.type == pygame.USEREVENT:
                    pygame.time.set_timer(pygame.USEREVENT, 300)
                    self.draw_all_screen()

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
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.done = True
                elif event.type == pygame.USEREVENT:
                    # Set speed
                    if not self.done:
                        keys_pressed = pygame.key.get_pressed()
                        if keys_pressed[pygame.K_DOWN]:
                            pygame.time.set_timer(pygame.USEREVENT, UIVariables.framerate * 1)
                        else:
                            pygame.time.set_timer(pygame.USEREVENT, UIVariables.framerate * 10)
                elif event.type == pygame.KEYDOWN:
                    keys_pressed = pygame.key.get_pressed()
                    if self.global_state.gui_input != '-':
                        self.global_state.gui_input = keys_pressed
            self.draw_all_screen()
            pygame.display.update()

        # Game over screen
        elif self.game_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.done = True
                elif event.type == pygame.USEREVENT:
                    pygame.time.set_timer(pygame.USEREVENT, 300)

                    self.draw_statistics()

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

    while True:
        # Update the gui depending on the framerate
        pygame.time.wait(UIVariables.framerate)
        draw_whole_screen()
