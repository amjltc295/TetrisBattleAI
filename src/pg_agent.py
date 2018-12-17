# -*- coding: utf-8 -*-
import math
import random
import sys
import os
import shutil
import numpy as np
import matplotlib
#import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from copy import deepcopy
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from engine import TetrisEngine

width, height = 10, 20 # standard tetris friends rules
engine = TetrisEngine(width, height)
eps = 10.**-8

# set up matplotlib
#is_ipython = 'inline' in matplotlib.get_backend()
#if is_ipython:
    #from IPython import display

#plt.ion()

# if gpu is to be used
use_cuda = torch.cuda.is_available()
if use_cuda:print("....Using Gpu...")
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
#Tensor = FloatTensor


######################################################################
# Replay Memory
# -------------
# -  ``Transition`` - a named tuple representing a single transition in
#    our environment
# -  ``ReplayMemory`` - a cyclic buffer of bounded size that holds the
#    transitions observed recently. It also implements a ``.sample()``
#    method for selecting a random batch of transitions for training.
#

Transition = namedtuple('Transition',
                        ('state', 'placement', 'other_score', 'next_state', 'reward'))


class PG(nn.Module):

    def __init__(self):
        super(PG, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.lin1 = nn.Linear(896, 256)
#         self.lin1 = nn.Linear(448, 256)
        self.head = nn.Linear(256 * 2, 1)

    def forward(self, x, placement):
        batch, ch, w, h = x.size()
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
#         print(x.shape)
        x = F.relu(self.lin1(x.view(batch, -1)))
        
        p = placement
        batch, ch, w, h = p.size()
        p = F.relu(self.bn1(self.conv1(p)))
        p = F.relu(self.bn2(self.conv2(p)))
        p = F.relu(self.lin1(p.view(batch, -1)))
        
        return self.head(torch.cat([x,p], dim=-1))
    
    

######################################################################
# Training
# --------
#
# Hyperparameters and utilities
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# This cell instantiates our model and its optimizer, and defines some
# utilities:
#
# -  ``Variable`` - this is a simple wrapper around
#    ``torch.autograd.Variable`` that will automatically send the data to
#    the GPU every time we construct a Variable.
# -  ``select_action`` - will select an action accordingly to an epsilon
#    greedy policy. Simply put, we'll sometimes use our model for choosing
#    the action, and sometimes we'll just sample one uniformly. The
#    probability of choosing a random action will start at ``EPS_START``
#    and will decay exponentially towards ``EPS_END``. ``EPS_DECAY``
#    controls the rate of the decay.
#

BATCH_SIZE = 512
GAMMA = 0.99
EPS_START = 1
EPS_END = 0.1
EPS_DECAY = 3000000
# GAMMA = 0.999
# EPS_START = 0.9
# EPS_END = 0.05
# EPS_DECAY = 200
CHECKPOINT_FILE = 'pg.pth.tar'



steps_done = 0

model = PG()
model.train()
print(model)

if use_cuda:
    model.cuda()

# loss_criterion = nn.MSELoss()
# optimizer = optim.RMSprop(model.parameters(), lr=.001)
optimizer = optim.Adam(model.parameters(), lr=.001)

def get_action_placement(engine):
    action_final_location_map = engine.get_valid_final_states(engine.shape, engine.anchor, engine.board)
    act_map = {k:v[2] for k,v in action_final_location_map.items()}
    return act_map

def entropy(act_prob):
    assert len(act_prob.shape) == 2
    entropy = torch.mean(-torch.log(act_prob + eps) * act_prob)
    return entropy
def discount_rewards(rewards):
    rewards = np.array(rewards, dtype=np.float)
    discounted_rewards = np.zeros_like(rewards)
    running_add = 0
    for t in reversed(range(0, rewards.size)):
        if t == rewards.size-1:
            discounted_rewards[t] = rewards[t]
            continue
        if rewards[t] != 0:
            running_add = 0
        running_add = running_add * GAMMA + rewards[t]
        discounted_rewards[t] = running_add
    rewards = discounted_rewards
#     rewards = (rewards - rewards.mean()) / (rewards.std())
    return rewards

def select_action(state, engine):
    act_map = get_action_placement(engine)
    with torch.no_grad():
        placements = [] 
        for k, v in act_map.items():
            placements.append(FloatTensor(v)[None, None,:,:])
        placements = torch.cat(placements, dim=0)
        states = FloatTensor(state)[None, None,:,:].repeat(len(act_map), 1, 1, 1)
        Q = model(states, placements).flatten()
        assert Q.shape == (len(act_map),)
        prob_action = F.softmax(Q, dim=0)
        act_idx = int(np.random.choice(len(act_map), 1, p=prob_action.cpu().numpy()))
        other_score = torch.sum(Q) - Q[act_idx]
    act = list(act_map.keys())[act_idx]
    placement = act_map[act]
    return act, placement, other_score
    

episode_durations = []


'''
def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.FloatTensor(episode_durations)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())
'''


######################################################################
# Training loop
# ^^^^^^^^^^^^^
#
# Finally, the code for training our model.
#
# Here, you can find an ``optimize_model`` function that performs a
# single step of the optimization. It first samples a batch, concatenates
# all the tensors into a single one, computes :math:`Q(s_t, a_t)` and
# :math:`V(s_{t+1}) = \max_a Q(s_{t+1}, a)`, and combines them into our
# loss. By defition we set :math:`V(s) = 0` if :math:`s` is a terminal
# state.


last_sync = 0
def optimize_model(episode):
    batch = Transition(*zip(*episode))


    state_batch = torch.cat([FloatTensor(s)[None, None,:,:] for s in batch.state])
    placement_batch = torch.cat([FloatTensor(s)[None, None,:,:] for s in batch.placement])
    other_score_batch = FloatTensor(batch.other_score)
    reward_batch = FloatTensor(discount_rewards(batch.reward))
    act_score = model(state_batch, placement_batch).flatten()
#     softmax probability
    act_prob = torch.exp(act_score) / torch.exp(act_score+other_score_batch)
#     loss = torch.mean(-torch.log(act_prob.flatten()) * reward_batch) - 0.01*entropy(act_prob)
    loss = torch.mean(-torch.log(act_prob + eps) * reward_batch)
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in model.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    return loss


def save_checkpoint(state, is_best, filename, best_name ):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_name)

def load_checkpoint(filename):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])
    cached_model.load_state_dict(checkpoint['state_dict'])
    try: # If these fail, its loading a supervised model
        optimizer.load_state_dict(checkpoint['optimizer'])
    except Exception as e:
        pass
    # Low chance of random action
    #steps_done = 10 * EPS_DECAY

    return checkpoint['epoch'], checkpoint['best_score']

if __name__ == '__main__':
    # Check if user specified to resume from a checkpoint
    start_epoch = 0
    best_score = float('-inf')
    if len(sys.argv) > 1 and sys.argv[1] == 'resume':
        if len(sys.argv) > 2:
            CHECKPOINT_FILE = sys.argv[2]
        if os.path.isfile(CHECKPOINT_FILE):
            print("=> loading checkpoint '{}'".format(CHECKPOINT_FILE))
            start_epoch, best_score = load_checkpoint(CHECKPOINT_FILE)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(CHECKPOINT_FILE, start_epoch))
        else:
            print("=> no checkpoint found at '{}'".format(CHECKPOINT_FILE))

    ######################################################################
    #
    # Below, you can find the main training loop. At the beginning we reset
    # the environment and initialize the ``state`` variable. Then, we sample
    # an action, execute it, observe the next screen and the reward (always
    # 1), and optimize our model once. When the episode ends (our model
    # fails), we restart the loop.
    
    f = open('log.out', 'w+')
    for i_episode in count(start_epoch):
        # Initialize the environment and state
        state = engine.clear()
        score = 0
        episode = []
        for t in count():
            # Select and perform an action
    
            action, placement, other_score = select_action(state, engine)
            # Observations
            last_state = state
            state, reward, done, cleared_lines = engine.step_to_final(action)
#             s = np.asarray(state)
#             s = np.swapaxes(s,1,0)
#             print(s)
#             print(engine)
#             print('reward ' , reward, done)
#             time.sleep(1)
            if done:
                state = None
                
            # Accumulate reward
            score += reward
#             if i_episode == 100:
#                 print(engine)
            # Store the transition in memory
            episode.append([last_state, placement, other_score, state, reward])
    
            # Perform one step of the optimization (on the target network)
            if done:
                loss = optimize_model(episode)
                # Train model
                if i_episode % 10 == 0:
    
                    log = 'epoch {0} score {1}'.format(i_episode, '%.2f' % score)
                    print(log)
                    f.write(log + '\n')
    
    
                    if loss:
                        print('loss: {:.2f}'.format(loss))
                # Checkpoint
                if i_episode % 100 == 0:
                    is_best = True if score > best_score else False
                    save_checkpoint({
                        'epoch' : i_episode,
                        'state_dict' : model.state_dict(),
                        'best_score' : best_score,
                        'optimizer' : optimizer.state_dict(),
                        }, is_best, filename='pg.pth.tar', best_name='pg_best.pth.tar')
                    best_score = score
                break
    

    f.close()
    print('Complete')
    #env.render(close=True)
    #env.close()
    #plt.ioff()
    #plt.show()
