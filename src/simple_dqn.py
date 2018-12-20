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
from heuristic import *
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
from engine import TetrisEngine
import fixed_policy_agent
from collections import deque
# width, height = 10, 20  # standard tetris friends rules
width, height = 10, 20  # standard tetris friends rules
engine = TetrisEngine(width, height)
action_n = 30

# if gpu is to be used
use_cuda = torch.cuda.is_available()
if use_cuda:
    print("....Using Gpu...")
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
# Tensor = FloatTensor


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
                        ('state', 'next_state', 'action', 'reward'))


class ReplayMemory(object):
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.old_memory = []
        self.memory = []
        self.position = 0
        self.prior = None
    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return self.prioritized_sweeping(batch_size)
    
    def update_memory(self):
        self.old_memory = self.memory.copy()
        self.calc_prior()
        
    def prioritized_sweeping(self, batch_size):
        return random.sample(self.memory, batch_size)
        if len(self.memory) < self.capacity:
            return random.sample(self.memory, batch_size)
        else:
#             print('prior')
            if self.prior is None:
                self.update_memory()
                
            idx = np.random.choice(np.arange(len(self.old_memory)), size=[batch_size,],p=self.prior)
            return np.array(self.old_memory)[idx]
#     def calc_prior(self):
# #         print('start', len(self.old_memory))
#         batch = Transition(*zip(*self.old_memory))
    
#         # Compute a mask of non-final states and concatenate the batch elements
#         non_final_mask = ByteTensor(tuple(map(lambda s: s is not None,
#                                               batch.next_state)))
    
#         state_batch = torch.cat(batch.state)
#         next_Q_batch = torch.cat(batch.next_max_Q)
#         reward_batch = torch.cat(batch.reward)
#         # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
#         # columns of actions taken

#         next_state_values = torch.zeros(state_batch.shape[0]).type(FloatTensor)
#         next_state_values[non_final_mask] = next_Q_batch[non_final_mask]

#         # Compute the expected Q values
#         expected_state_values = (next_state_values * GAMMA) + reward_batch
#         prior = state_values.flatten() - expected_state_values
#         prior = torch.abs(prior)
#         prior = F.softmax(prior, dim=0)
#         self.prior = prior.cpu().numpy()
# #         print('end',self.prior.shape,self.prior[:100])

    def __len__(self):
        return len(self.memory)

    

class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.cnn = CNN_lay()
        
        # self.bn3 = nn.BatchNorm2d(32)
        # self.rnn = nn.LSTM(448, 240)
        self.lin1 = nn.Linear(640, 64)
        self.Q_lin = nn.Linear(64, action_n)
        
    def forward(self, x):
        batch,_,_,_ = x.shape
        x = self.cnn(x)
        x = x.view(batch, -1)
        
        x = F.relu(self.lin1(x))
        
        Q = self.Q_lin(x)
        return Q

class CNN_lay(nn.Module):

    def __init__(self):
        super(CNN_lay, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        self.conv_collapse = nn.Conv2d(32, 64, kernel_size=(1,20), stride=1)
        self.bn4 = nn.BatchNorm2d(64)
        
        self.conv5 = nn.Conv2d(64, 64, kernel_size=(3,1), stride=1, padding=(1,0))
        self.bn5 = nn.BatchNorm2d(64)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn4(self.conv_collapse(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        return x

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

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 1
EPS_END = 0.2
EPS_DECAY = 30000
# GAMMA = 0.999
# EPS_START = 0.9
# EPS_END = 0.05
# EPS_DECAY = 200



steps_done = 0

model = DQN()
cached_model = DQN()
cached_model.eval()
for p in cached_model.parameters():
    p.requires_grad_(False)
print(model)

if use_cuda:
    model.cuda()
    cached_model.cuda()

# loss_criterion = nn.MSELoss()
# optimizer = optim.RMSprop(model.parameters(), lr=.001)
optimizer = optim.Adam(model.parameters(), lr=.001)
memory = ReplayMemory(20000)

def get_max_Q(model, state, engine, shape, anchor, board):
    action_final_location_map = engine.get_valid_final_states(shape, anchor, board)
    act_pairs = [ (k,v[2]) for k,v in action_final_location_map.items()]
    with torch.no_grad():
        states = FloatTensor(state)[None, None,:,:]
        Q = model(states)
        assert Q.shape[1] == len(act_pairs)
        topv, topi = torch.topk(Q, dim=1, k=1)
        act, placement = act_pairs[topi.item()]
    return act, placement, topv.item()

def select_action(model, state, engine, shape, anchor, board):
    sample = random.random()
    eps_threshold = cal_eps()
    if sample > eps_threshold:
        act, placement, Q = get_max_Q(model, state, engine, shape, anchor, board)
        return act, placement
                
    else:
        action_final_location_map = engine.get_valid_final_states(shape, anchor, board)
        act_pairs = [ (k,v[2]) for k,v in action_final_location_map.items()]
#         if random.random() <= 0.3:
#             idx = random.randrange(len(act_pairs))
#             act, placement = act_pairs[idx]
#             return act, placement
#         else:
        return fixed_policy_agent.select_action(engine, engine.shape, engine.anchor, engine.board)
            
        
def cal_eps():
    global steps_done
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    return eps_threshold




def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    non_final_mask = ByteTensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)))
    non_final_next_states = torch.cat([FloatTensor(s)[None, None,:,:] for s in batch.next_state
                                                if s is not None], dim=0)
    state_batch = torch.cat([FloatTensor(s)[None, None,:,:] for s in batch.state], dim=0)
    action_batch = LongTensor(batch.action).unsqueeze(1)
    reward_batch = FloatTensor(batch.reward).unsqueeze(1)
    Q = torch.gather(model(state_batch), 1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    next_Q = torch.zeros_like(Q)
    with torch.no_grad():
        next_Q[non_final_mask],_ = torch.topk(model(non_final_next_states), dim=1, k=1)
    target = (next_Q * GAMMA) + reward_batch

    loss = F.smooth_l1_loss(target, Q)

#     loss = loss_criterion(state_action_values.squeeze(1), expected_state_action_values)

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
        # memory = checkpoint['memory']
    except Exception:
        pass
    # Low chance of random action
    # steps_done = 10 * EPS_DECAY

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
    
    f = open('Q_log.out', 'w+')
    score_q = deque(maxlen = 100)
    for i_episode in count(start_epoch):
        # Initialize the environment and state
        state = engine.clear()
        state,_,_,_ = engine.step(6)
        
        score = 0
        reward_sum = 0
        model.eval()
        for t in count():
            # Select and perform an action
            action_final_location_map = engine.get_valid_final_states(engine.shape, engine.anchor, engine.board)
            action, placement = select_action(model, state, engine, engine.shape, engine.anchor, engine.board)
            act_idx = list(action_final_location_map.keys()).index(action)
            # Observations
            last_state = state
            
            state, reward, done, cleared_lines = engine.step_to_final(action)
            
            if done:
                data = (last_state, None, act_idx, reward)
            else:
                data = (last_state, state, act_idx, reward)
            
            memory.push(*data)
            # Accumulate reward
#             score += reward
            score += cleared_lines
            reward_sum += reward

            
            # Perform one step of the optimization (on the target network)
            if done or t >= 200:
                model.train()
                loss = optimize_model()
                # Train model
                score_q.append(score)
                if i_episode % 10 == 0:
    
                    log = 'epoch {0} score {1} rewards {3} score_mean {4} eps {2}'.format(i_episode, '%.2f' % score, cal_eps(),
                                                                         '%.2f' % reward_sum, '%.2f' % np.mean(score_q))
                    print(log)
                    f.write(log + '\n')
    
                
                    if loss:
                        print('loss: {:.2f}'.format(loss))
                # Checkpoint
                if i_episode % 100 == 0:
                    is_best = True if score > best_score else False
                    save_checkpoint({
                        'epoch' : i_episode,
                        'state_dict' : cached_model.state_dict(),
                        'best_score' : best_score,
                        'optimizer' : optimizer.state_dict(),
#                         'memory' : memory
                        }, is_best, filename='Q.pth.tar', best_name='Q_best.pth.tar')
                    best_score = score
#                 copy cached model
                if i_episode % 50 == 0 and  i_episode > 0:
                    cached_model.load_state_dict(model.state_dict())
#                     memory.update_memory()
                break
    

    f.close()
    print('Complete')
