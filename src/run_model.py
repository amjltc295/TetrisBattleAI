import sys
import os
import torch
import time
from engine import TetrisEngine
from dqn_agent import DQN
from dqn_agent import select_action
from torch.autograd import Variable
import fixed_policy_agent
use_cuda = torch.cuda.is_available()

FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor

width, height = 10, 20  # standard tetris friends rules
engine = TetrisEngine(width, height)


def load_model(filename):
    model = DQN()
    if use_cuda:
        model.cuda()
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])

    return model


def run(model):
    state = FloatTensor(engine.clear()[None, None, :, :])
    score = 0
    model.eval()
    with torch.no_grad():
        while True:

            action, placement = select_action(model, state, engine, engine.shape, engine.anchor, engine.board)
            # action, placement = fixed_policy_agent.select_action(engine, engine.shape, engine.anchor, engine.board)

            state, reward, done, cleared_lines = engine.step_to_final(action)
            state = FloatTensor(state[None,None,:,:])

            # Accumulate reward
            score += int(reward)

            print(engine)
            print('score : %d' % score)
            # print(action)
            time.sleep(.1)

            if done:
                print('score {0}'.format(score))
                break


if len(sys.argv) <= 1:
    print('specify a filename to load the model')
    sys.exit(1)

if __name__ == '__main__':
    filename = sys.argv[1]
    if os.path.isfile(filename):
        print("=> loading model '{}'".format(filename))
        model = load_model(filename).eval()
        run(model)
    else:
        print("=> no file found at '{}'".format(filename))
