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

use_cuda = torch.cuda.is_available()
if use_cuda:print("....Using Gpu...")
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor

Transition = namedtuple('Transition',
                        ('state', 'action', 'shape', 'anchor', 'board', 'reward'))


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
    
    

GAMMA = 0.99
CHECKPOINT_FILE = 'pg.pth.tar'




model = PG()
print(model)

if use_cuda:
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr=.001)



def entropy(act_prob):
    assert len(act_prob.shape) == 1
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
    rewards = (rewards - rewards.mean()) / (rewards.std())
    return rewards

def get_action_probability(model, state, act_pairs):
    
    placements = torch.cat([ FloatTensor(v)[None, None,:,:] for k, v in act_pairs ], dim=0)
    states = FloatTensor(state)[None, None,:,:].repeat(len(act_pairs), 1, 1, 1)
    act_score = model(states, placements).flatten()
    assert act_score.shape == (len(act_pairs),)
    act_prob = F.softmax(act_score, dim=0)
    return act_prob

def select_action(model, state, engine, shape, anchor, board):
    model.eval()
    action_final_location_map = engine.get_valid_final_states(shape, anchor, board)
    act_pairs = [k:v[2] for k,v in action_final_location_map.items()]
    
    with torch.no_grad():
        act_prob = get_action_probability(model, state, act_pairs)
        
    act_idx = int(np.random.choice(len(act_prob), 1, p=act_prob.cpu().numpy()))
    act, placement = act_pairs[act_idx]
    return act, placement

def optimize_model(model, engine, episode):
    batch = Transition(*zip(*episode))

    model.train()
    loss = 0
    state_batch = torch.cat([FloatTensor(s)[None, None,:,:] for s in batch.state])
    rewards = discount_rewards(batch.reward)
    for i in range(len(batch)):
        state = FloatTensor(batch.state[i])[None, None,:,:]
        shape = batch.shape[i]
        action = batch.action[i]
        anchor = batch.anchor[i]
        board = batch.board[i]
        act_pairs = [k:v[2] for k, v in engine.get_valid_final_states(shape, anchor, board).items()]
        act_prob = get_action_probability(model, state, act_pairs)
        action_idx = [k for k, v in act_pairs].index(action)
        loss += -torch.log(act_prob[action_idx] + eps) * rewards[i] - entropy(act_prob)
        
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
        reward_sum = 0
        for t in count():
            # Select and perform an action
    
            action, placement, other_score = select_action(state, engine)
            # Observations
            last_state = state
            state, reward, done, cleared_lines = engine.step_to_final(action)
            if done:
                state = None
                
            # Accumulate reward
#             score += reward
            score += cleared_lines
            reward_sum += reward
#              ('state', 'action', 'shape', 'anchor', 'board', 'reward'))
            episode.append([last_state, placement, other_score, state, reward])
    
            # Perform one step of the optimization (on the target network)
            if done:
                loss = optimize_model(model, engine, episode)
                # Train model
                if i_episode % 10 == 0:
    
                    log = 'epoch {0} score {1} rewards {2}'.format(i_episode, '%.2f' % score, '%.2f' % reward_sum)
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
