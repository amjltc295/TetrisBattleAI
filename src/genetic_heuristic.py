import math


def get_hole_stack_area(state):
    width, height = state.shape
    n_hole = 0
    for x in range(width-4):
        pre_occupied = True
        for y in reversed(range(height)):
            if not pre_occupied and state[x, y] == 1:
                n_hole += 1
            pre_occupied = state[x, y] == 1
    # print("Holes in stack area: ", n_hole)
    return n_hole


def get_hole_clean_area(state):
    width, height = state.shape
    n_hole = 0
    for x in range(width-4, width):
        pre_occupied = True
        for y in reversed(range(height)):
            if not pre_occupied and state[x, y] == 1:
                n_hole += 1
            pre_occupied = state[x, y] == 1
    # print("Holes in clean area: ", n_hole)
    return n_hole


def get_height(state, x):
    width, height = state.shape
    for y in range(height):
        if state[x, y] == 1:
            return height-y
    return 0


def max_height_stack_area(state):
    width, height = state.shape
    max_height = 0
    for x in range(width-4):
        max_height = max(max_height, get_height(state, x))
    # print("Max height stack area: ", max_height)
    return max_height


def max_height_clean_area(state):
    width, height = state.shape
    max_height = 0
    for x in range(width-4, width):
        max_height = max(max_height, get_height(state, x))
    # print("Max height clean area: ", max_height)
    return max_height


def aggregate_height_stack_area(state):
    width, height = state.shape
    n_height = 0
    for x in range(width-4):
        n_height += get_height(state, x)
    return n_height


def aggregate_height_clean_area(state):
    width, height = state.shape
    n_height = 0
    for x in range(width-4, width):
        n_height += get_height(state, x)
    return n_height


def bumpiness(state):
    width, height = state.shape
    ret = 0
    pre_height = 0
    for x in range(width):
        h = get_height(state, x)
        if x == 0:
            diff = 0
        else:
            diff = abs(h - pre_height)
        pre_height = h
        ret += diff
    return ret


def complete_line(state):
    width, height = state.shape
    n_line = 0
    for y in range(height):
        if (state[:, y] == 1).all():
            n_line += 1
    return n_line


def rect_func(x, boarder=16):
    if x <= boarder:
        return x
    else:
        return -x


def gen_heuristic(state, dict_genes):
    holes_value = dict_genes['holes_stack_area'] * (get_hole_stack_area(state)**2) + \
                  dict_genes['holes_clean_area'] * (get_hole_clean_area(state)**2)
    max_height_value = dict_genes['height_stack_area'] * rect_func(max_height_stack_area(
        state)) + dict_genes['height_clean_area'] * max_height_clean_area(state)
    aggregation_value = dict_genes['aggregation_stack_area'] * aggregate_height_stack_area(state)
    clear_lines = dict_genes['clear_lines'] * (math.exp(complete_line(state)**2/2) - 1)
    bumpiness_value = dict_genes['bumpiness'] * bumpiness(state)
    return holes_value + max_height_value + aggregation_value + clear_lines + bumpiness_value
