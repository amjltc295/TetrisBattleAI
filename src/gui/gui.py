# Based on:
# PYTRIS™ Copyright (c) 2017 Jason Kim All Rights Reserved.

import pygame
import operator
from src.gui.mino import *
from random import *
from pygame.locals import *
from src.global_engine import *
from src.engine import *


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
        self.screen_height = 374
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.time.set_timer(pygame.USEREVENT, UIVariables.framerate * 10)
        pygame.display.set_caption("Make Tetris Great Again")

    # Draw block
    def draw_block(self, x, y, color):
        pygame.draw.rect(
            self.screen,
            color,
            Rect(x, y, UIVariables.block_size, UIVariables.block_size)
        )
        pygame.draw.rect(
            self.screen,
            UIVariables.grey_1,
            Rect(x, y, UIVariables.block_size, UIVariables.block_size),
            1
        )

    # Draw game screen
    def draw_board(self):
        # Background gray
        self.screen.fill(UIVariables.grey_1)

        # Draw sidebar 1
        pygame.draw.rect(
            self.screen,
            UIVariables.white,
            Rect(0, 0, 96, 374)
        )
        # Draw sidebar 2
        pygame.draw.rect(
            self.screen,
            UIVariables.white,
            Rect(504, 0, 96, 374)
        )

        # Draw next mino 1
        grid_n1 = tetrimino.mino_map[self.global_state.engines[0].next_shape - 1][0]
        for i in range(4):
            for j in range(4):
                dx = 20 + UIVariables.block_size * j
                dy = 140 + UIVariables.block_size * i
                if grid_n1[i][j] != 0:
                    pygame.draw.rect(
                        self.screen,
                        UIVariables.t_color[grid_n1[i][j]],
                        Rect(dx, dy, UIVariables.block_size, UIVariables.block_size)
                    )
        # Draw next mino 2
        grid_n2 = tetrimino.mino_map[self.global_state.engines[1].next_shape - 1][0]
        for i in range(4):
            for j in range(4):
                dx = 520 + UIVariables.block_size * j
                dy = 140 + UIVariables.block_size * i
                if grid_n2[i][j] != 0:
                    pygame.draw.rect(
                        self.screen,
                        UIVariables.t_color[grid_n2[i][j]],
                        Rect(dx, dy, UIVariables.block_size, UIVariables.block_size)
                    )

        # Draw hold mino 1
        grid_h1 = tetrimino.mino_map[self.global_state.engines[0].hold_shape - 1][0]
        if self.global_state.engines[0].holded:
            for i in range(4):
                for j in range(4):
                    dx = 20 + UIVariables.block_size * j
                    dy = 50 + UIVariables.block_size * i
                    if grid_h1[i][j] != 0:
                        pygame.draw.rect(
                            self.screen,
                            UIVariables.t_color[grid_h1[i][j]],
                            Rect(dx, dy, UIVariables.block_size, UIVariables.block_size)
                        )
        # Draw hold mino 2
        grid_h2 = tetrimino.mino_map[self.global_state.engines[1].hold_shape - 1][0]
        if self.global_state.engines[1].holded:
            for i in range(4):
                for j in range(4):
                    dx = 20 + UIVariables.block_size * j
                    dy = 50 + UIVariables.block_size * i
                    if grid_h2[i][j] != 0:
                        pygame.draw.rect(
                            self.screen,
                            UIVariables.t_color[grid_h2[i][j]],
                            Rect(dx, dy, UIVariables.block_size, UIVariables.block_size)
                        )

        # Draw texts
        text_hold = UIVariables.h5.render("HOLD", 1, UIVariables.black)
        text_next = UIVariables.h5.render("NEXT", 1, UIVariables.black)
        text_score = UIVariables.h5.render("SCORE", 1, UIVariables.black)
        score1_value = UIVariables.h4.render(str(self.global_state.engines[0].score), 1, UIVariables.black)
        score2_value = UIVariables.h4.render(str(self.global_state.engines[1].score), 1, UIVariables.black)
        text_lines = UIVariables.h5.render("Total cleared lines", 1, UIVariables.black)
        lines1_value = UIVariables.h4.render(str(self.global_state.engines[0].total_cleared_lines), 1, UIVariables.black)
        lines2_value = UIVariables.h4.render(str(self.global_state.engines[1].total_cleared_lines), 1, UIVariables.black)
        text_ko = UIVariables.h5.render("KO's", 1, UIVariables.black)
        ko1_value = UIVariables.h4.render(str(self.global_state.engines[0].n_deaths), 1, UIVariables.black)
        ko2_value = UIVariables.h4.render(str(self.global_state.engines[1].n_deaths), 1, UIVariables.black)

        # Place texts
        self.screen.blit(text_hold, (15, 14))
        self.screen.blit(text_hold, (515, 14))
        self.screen.blit(text_next, (15, 104))
        self.screen.blit(text_next, (515, 104))
        self.screen.blit(text_score, (15, 194))
        self.screen.blit(text_score, (515, 194))
        self.screen.blit(score1_value, (20, 210))
        self.screen.blit(score2_value, (520, 210))
        self.screen.blit(text_lines, (15, 254))
        self.screen.blit(text_lines, (515, 254))
        self.screen.blit(lines1_value, (20, 270))
        self.screen.blit(lines2_value, (520, 270))
        self.screen.blit(text_ko, (15, 314))
        self.screen.blit(text_ko, (515, 314))
        self.screen.blit(ko1_value, (20, 330))
        self.screen.blit(ko2_value, (520, 330))

        # Draw board 1
        for x in range(self.global_state.width):
            for y in range(self.global_state.height):
                dx = 17 + UIVariables.block_size * x
                dy = 17 + UIVariables.block_size * y
                self.draw_block(dx, dy, UIVariables.t_color[self.global_state.engines[0].board[x][y + 1]])
        # Draw board 2
        for x in range(self.global_state.width):
            for y in range(self.global_state.height):
                dx = 317 + UIVariables.block_size * x
                dy = 317 + UIVariables.block_size * y
                self.draw_block(dx, dy, UIVariables.t_color[self.global_state.engines[1].board[x][y + 1]])

    # TODO
    def draw_statistics(self):
                return

    def draw_screen(self):
        # Pause screen
        if self.pause:
            for event in pygame.event.get():
                if event.type == QUIT:
                    self.done = True
                elif event.type == USEREVENT:
                    pygame.time.set_timer(pygame.USEREVENT, 300)
                    self.draw_board()

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
                    if event.key == K_ESCAPE:
                        self.pause = False
                        pygame.time.set_timer(pygame.USEREVENT, 1)

        # Game screen
        elif self.start:
            for event in pygame.event.get():
                if event.type == QUIT:
                    self.done = True
                elif event.type == USEREVENT:
                    # Set speed
                    if not self.done:
                        keys_pressed = pygame.key.get_pressed()
                        if keys_pressed[K_DOWN]:
                            pygame.time.set_timer(pygame.USEREVENT, UIVariables.framerate * 1)
                        else:
                            pygame.time.set_timer(pygame.USEREVENT, UIVariables.framerate * 10)
                    # Draw board
                    self.draw_board()
                elif event.type == pygame.KEYDOWN:
                    keys_pressed = pygame.key.get_pressed()

            pygame.display.update()

        # Game over screen
        elif self.game_over:
            for event in pygame.event.get():
                if event.type == QUIT:
                    self.done = True
                elif event.type == USEREVENT:
                    pygame.time.set_timer(pygame.USEREVENT, 300)

                    # TODO: show statistics screen
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
                    keys_pressed = pygame.key.get_pressed()


while not done:
    # Update the gui depending on the framerate
    pygame.time.wait(UIVariables.framerate)
pygame.quit()

