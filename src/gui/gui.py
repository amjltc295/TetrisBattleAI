# Based on:
# PYTRIS™ Copyright (c) 2017 Jason Kim All Rights Reserved.

import pygame
import operator
from src.gui.mino import *
from random import *
from pygame.locals import *
from src.global_engine import *
from src.engine import *


# Initial values
blink = False
pause = False
done = False

name_location = 0
name = [65, 65, 65]

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
    def draw_board(self, next, hold, score, level, goal):
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

while not done:
    # Update the gui depending on the framerate
    pygame.time.wait(UIVariables.framerate)

    # Pause screen
    if pause:
        for event in pygame.event.get():
            if event.type == QUIT:
                done = True
            elif event.type == USEREVENT:
                pygame.time.set_timer(pygame.USEREVENT, 300)
                draw_board(next_mino, hold_mino, score, level, goal)

                pause_text = UIVariables.h2_b.render("PAUSED", 1, UIVariables.white)
                pause_start = UIVariables.h5.render("Press esc to continue", 1, UIVariables.white)

                screen.blit(pause_text, (43, 100))
                if blink:
                    screen.blit(pause_start, (40, 160))
                    blink = False
                else:
                    blink = True
                pygame.display.update()
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    pause = False
                    pygame.time.set_timer(pygame.USEREVENT, 1)

    # Game screen
    elif start:
        for event in pygame.event.get():
            if event.type == QUIT:
                done = True
            elif event.type == USEREVENT:
                # Set speed
                if not game_over:
                    keys_pressed = pygame.key.get_pressed()
                    if keys_pressed[K_DOWN]:
                        pygame.time.set_timer(pygame.USEREVENT, framerate * 1)
                    else:
                        pygame.time.set_timer(pygame.USEREVENT, framerate * 10)

                # Draw a mino
                draw_mino(dx, dy, mino, rotation)
                draw_board(next_mino, hold_mino, score, level, goal)

                # Erase a mino
                if not game_over:
                    erase_mino(dx, dy, mino, rotation)

                # Move mino down
                if not is_bottom(dx, dy, mino, rotation):
                    dy += 1

                # Create new mino
                else:
                    if hard_drop or bottom_count == 6:
                        hard_drop = False
                        bottom_count = 0
                        score += 10 * level
                        draw_mino(dx, dy, mino, rotation)
                        draw_board(next_mino, hold_mino, score, level, goal)
                        if is_stackable(next_mino):
                            mino = next_mino
                            next_mino = randint(1, 7)
                            dx, dy = 3, 0
                            rotation = 0
                            hold = False
                        else:
                            start = False
                            game_over = True
                            pygame.time.set_timer(pygame.USEREVENT, 1)
                    else:
                        bottom_count += 1

                # Erase line
                erase_count = 0
                for j in range(21):
                    is_full = True
                    for i in range(10):
                        if matrix[i][j] == 0:
                            is_full = False
                    if is_full:
                        erase_count += 1
                        k = j
                        while k > 0:
                            for i in range(10):
                                matrix[i][k] = matrix[i][k - 1]
                            k -= 1
                if erase_count == 1:
                    UIVariables.single_sound.play()
                    score += 50 * level
                elif erase_count == 2:
                    UIVariables.double_sound.play()
                    score += 150 * level
                elif erase_count == 3:
                    UIVariables.triple_sound.play()
                    score += 350 * level
                elif erase_count == 4:
                    UIVariables.tetris_sound.play()
                    score += 1000 * level

                # Increase level
                goal -= erase_count
                if goal < 1 and level < 15:
                    level += 1
                    goal += level * 5
                    framerate = int(framerate * 0.8)

            elif event.type == KEYDOWN:
                erase_mino(dx, dy, mino, rotation)
                if event.key == K_ESCAPE:
                    UIVariables.click_sound.play()
                    pause = True
                # Hard drop
                elif event.key == K_SPACE:
                    UIVariables.drop_sound.play()
                    while not is_bottom(dx, dy, mino, rotation):
                        dy += 1
                    hard_drop = True
                    pygame.time.set_timer(pygame.USEREVENT, 1)
                    draw_mino(dx, dy, mino, rotation)
                    draw_board(next_mino, hold_mino, score, level, goal)
                # Hold
                elif event.key == K_LSHIFT or event.key == K_c:
                    if hold == False:
                        UIVariables.move_sound.play()
                        if hold_mino == -1:
                            hold_mino = mino
                            mino = next_mino
                            next_mino = randint(1, 7)
                        else:
                            hold_mino, mino = mino, hold_mino
                        dx, dy = 3, 0
                        rotation = 0
                        hold = True
                    draw_mino(dx, dy, mino, rotation)
                    draw_board(next_mino, hold_mino, score, level, goal)
                # Turn right
                elif event.key == K_UP or event.key == K_x:
                    if is_turnable_r(dx, dy, mino, rotation):
                        UIVariables.move_sound.play()
                        rotation += 1
                    # Kick
                    elif is_turnable_r(dx, dy - 1, mino, rotation):
                        UIVariables.move_sound.play()
                        dy -= 1
                        rotation += 1
                    elif is_turnable_r(dx + 1, dy, mino, rotation):
                        UIVariables.move_sound.play()
                        dx += 1
                        rotation += 1
                    elif is_turnable_r(dx - 1, dy, mino, rotation):
                        UIVariables.move_sound.play()
                        dx -= 1
                        rotation += 1
                    elif is_turnable_r(dx, dy - 2, mino, rotation):
                        UIVariables.move_sound.play()
                        dy -= 2
                        rotation += 1
                    elif is_turnable_r(dx + 2, dy, mino, rotation):
                        UIVariables.move_sound.play()
                        dx += 2
                        rotation += 1
                    elif is_turnable_r(dx - 2, dy, mino, rotation):
                        UIVariables.move_sound.play()
                        dx -= 2
                        rotation += 1
                    if rotation == 4:
                        rotation = 0
                    draw_mino(dx, dy, mino, rotation)
                    draw_board(next_mino, hold_mino, score, level, goal)
                # Turn left
                elif event.key == K_z or event.key == K_LCTRL:
                    if is_turnable_l(dx, dy, mino, rotation):
                        UIVariables.move_sound.play()
                        rotation -= 1
                    # Kick
                    elif is_turnable_l(dx, dy - 1, mino, rotation):
                        UIVariables.move_sound.play()
                        dy -= 1
                        rotation -= 1
                    elif is_turnable_l(dx + 1, dy, mino, rotation):
                        UIVariables.move_sound.play()
                        dx += 1
                        rotation -= 1
                    elif is_turnable_l(dx - 1, dy, mino, rotation):
                        UIVariables.move_sound.play()
                        dx -= 1
                        rotation -= 1
                    elif is_turnable_l(dx, dy - 2, mino, rotation):
                        UIVariables.move_sound.play()
                        dy -= 2
                        rotation += 1
                    elif is_turnable_l(dx + 2, dy, mino, rotation):
                        UIVariables.move_sound.play()
                        dx += 2
                        rotation += 1
                    elif is_turnable_l(dx - 2, dy, mino, rotation):
                        UIVariables.move_sound.play()
                        dx -= 2
                    if rotation == -1:
                        rotation = 3
                    draw_mino(dx, dy, mino, rotation)
                    draw_board(next_mino, hold_mino, score, level, goal)
                # Move left
                elif event.key == K_LEFT:
                    if not is_leftedge(dx, dy, mino, rotation):
                        UIVariables.move_sound.play()
                        dx -= 1
                    draw_mino(dx, dy, mino, rotation)
                    draw_board(next_mino, hold_mino, score, level, goal)
                # Move right
                elif event.key == K_RIGHT:
                    if not is_rightedge(dx, dy, mino, rotation):
                        UIVariables.move_sound.play()
                        dx += 1
                    draw_mino(dx, dy, mino, rotation)
                    draw_board(next_mino, hold_mino, score, level, goal)

        pygame.display.update()

    # Game over screen
    elif game_over:
        for event in pygame.event.get():
            if event.type == QUIT:
                done = True
            elif event.type == USEREVENT:
                pygame.time.set_timer(pygame.USEREVENT, 300)

                # TODO: show statistics screen
                draw_board(next_mino, hold_mino, score, level, goal)

                # Show "Game over"
                over_text_1 = UIVariables.h2_b.render("GAME", 1, UIVariables.white)
                over_text_2 = UIVariables.h2_b.render("OVER", 1, UIVariables.white)
                screen.blit(over_text_1, (58, 75))
                screen.blit(over_text_2, (62, 105))

                over_start = UIVariables.h5.render("Press return to exit", 1, UIVariables.white)
                if blink:
                    screen.blit(over_start, (32, 195))
                    blink = False
                else:
                    blink = True
                pygame.display.update()
            elif event.type == KEYDOWN:
                if event.key == K_RETURN:
                    UIVariables.click_sound.play()

                    outfile = open('leaderboard.txt','a')
                    outfile.write(chr(name[0]) + chr(name[1]) + chr(name[2]) + ' ' + str(score) + '\n')
                    outfile.close()

                    game_over = False
                    hold = False
                    dx, dy = 3, 0
                    rotation = 0
                    mino = randint(1, 7)
                    next_mino = randint(1, 7)
                    hold_mino = -1
                    framerate = 30
                    score = 0
                    score = 0
                    level = 1
                    goal = level * 5
                    bottom_count = 0
                    hard_drop = False
                    name_location = 0
                    name = [65, 65, 65]
                    matrix = [[0 for y in range(height + 1)] for x in range(width)]

                    with open('leaderboard.txt') as f:
                        lines = f.readlines()
                    lines = [line.rstrip('\n') for line in open('leaderboard.txt')]

                    leaders = {'AAA': 0, 'BBB': 0, 'CCC': 0}
                    for i in lines:
                        leaders[i.split(' ')[0]] = int(i.split(' ')[1])
                    leaders = sorted(leaders.items(), key=operator.itemgetter(1), reverse=True)

                    pygame.time.set_timer(pygame.USEREVENT, 1)
                elif event.key == K_RIGHT:
                    if name_location != 2:
                        name_location += 1
                    else:
                        name_location = 0
                    pygame.time.set_timer(pygame.USEREVENT, 1)
                elif event.key == K_LEFT:
                    if name_location != 0:
                        name_location -= 1
                    else:
                        name_location = 2
                    pygame.time.set_timer(pygame.USEREVENT, 1)
                elif event.key == K_UP:
                    UIVariables.click_sound.play()
                    if name[name_location] != 90:
                        name[name_location] += 1
                    else:
                        name[name_location] = 65
                    pygame.time.set_timer(pygame.USEREVENT, 1)
                elif event.key == K_DOWN:
                    UIVariables.click_sound.play()
                    if name[name_location] != 65:
                        name[name_location] -= 1
                    else:
                        name[name_location] = 90
                    pygame.time.set_timer(pygame.USEREVENT, 1)

# TODO: When game is done
pygame.quit()