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
from collections import deque
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
    
    

GAMMA = 0.9
CHECKPOINT_FILE = 'pg.pth.tar'




model = PG()
print(model)

if use_cuda:
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr=.001)



def entropy(act_prob):
    assert len(act_prob.shape) == 1
    entropy = torch.sum(-torch.log(act_prob + eps) * act_prob)
    return entropy

def discount_rewards(rewards):
    rewards = np.array(rewards, dtype=np.float)
    discounted_rewards = np.zeros_like(rewards)
    running_add = 0
    for t in reversed(range(0, len(rewards))):
        if rewards[t] != 0:
            running_add = 0
        running_add = running_add * GAMMA + rewards[t]
        discounted_rewards[t] = running_add
    rewards = discounted_rewards
#     rewards = (rewards - rewards.mean()) / (rewards.std())
    return rewards



def get_action_probability(model, state, act_pairs):
    
    placements = torch.cat([ FloatTensor(v)[None, None,:,:] for k, v in act_pairs ], dim=0)
    states = FloatTensor(state)[None, None,:,:].repeat(len(act_pairs), 1, 1, 1)
    act_score = model(states, placements).flatten()
    assert act_score.shape == (len(act_pairs),)
    act_prob = F.softmax(act_score, dim=0)
    return act_prob




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
    checkpoint = torch.load('./pg_best.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])
    r_q = deque(maxlen = 1000)
    for i in range(100):
        r_q.append(-90)
    model.train()
    f = open('pg_original_reward.out', 'w+')
    for i_episode in count(start_epoch):
        # Initialize the environment and state
        state = engine.clear()
        score = 0
        rewards = []
        entropy_loss = 0
        act_prob_list = []
        for t in count():
            # Select and perform an action
            action_final_location_map = engine.get_valid_final_states(engine.shape, engine.anchor, engine.board)
            act_pairs = [ (k, v[2]) for k,v in action_final_location_map.items()]

            act_prob = get_action_probability(model, state, act_pairs)
            act_idx = int(np.random.choice(len(act_prob), 1, p=act_prob.cpu().detach().numpy()))
            act_prob_list.append(act_prob[act_idx].unsqueeze(0))
            entropy_loss += -entropy(act_prob)
            act, placement = act_pairs[act_idx]

            # Observations
            state, reward, done, cleared_lines = engine.step_to_final(act)
            # Accumulate reward
            score += cleared_lines
            rewards.append(reward)
            
        
            if done:
#                 Optimize the model
#                 discounted_rewards = FloatTensor(discount_rewards(rewards))
                baseline = np.mean(r_q)
                R = sum(rewards)
                discounted_rewards = [0]*len(rewards)
                discounted_rewards[-1] = R - baseline
                discounted_rewards = FloatTensor(discount_rewards(discounted_rewards))
#                 print(discounted_rewards)
                act_probs = torch.cat(act_prob_list, dim=0)
                loss = torch.sum(-torch.log(act_probs + eps) * discounted_rewards) + 0.01*entropy_loss
                optimizer.zero_grad()
                loss.backward()
                for param in model.parameters():
                    param.grad.data.clamp_(-1, 1)
                optimizer.step()
                r_q.append(R)
                log = 'epoch {0} score {1} rewards {2} baseline {3}'.format(i_episode, '%.2f' % score, '%.2f' % R, '%.2f' % baseline)
                f.write(log + '\n')
                # Train model
                if i_episode % 10 == 0:
    
                    print(log)
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
