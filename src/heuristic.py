def get_hole(state):
    width, height = state.shape
    n_hole = 0
    for x in range(width):
        pre_occupied = True
        for y in reversed(range(height)):
            if not pre_occupied and state[x,y] == 1:
                n_hole += 1
            pre_occupied = state[x,y] == 1
    return n_hole

def get_height(state, x):
    width, height = state.shape
    for y in range(height):
        if state[x,y] == 1:
            return height-y
    return 0
def max_height(state):
    width, height = state.shape
    max_height = 0
    for x in range(width):
        max_height = max(max_height, get_height(state,x))
    return max_height

def aggregate_height(state):
    width, height = state.shape
    n_height = 0
    for x in range(width):
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
        if (state[:,y] == 1).all():
            n_line+=1
    return n_line
    
def heuristic_fn(state, cleared_lines):
#     import numpy as np
#     s = np.asarray(state)
#     s = np.swapaxes(s, 1,0)
#     print(s)
#     print(aggregate_height(state),complete_line(state),get_hole(state),bumpiness(state))
#     wef
    return -0.51*aggregate_height(state) + 0.76*cleared_lines - 0.36*get_hole(state) - 0.18*bumpiness(state)
#     return -0.51*aggregate_height(state) + 10.*complete_line(state) - 0.36*get_hole(state) - 0.18*bumpiness(state) - 0.3*max_height(state)
    
    