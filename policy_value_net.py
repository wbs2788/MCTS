'''
Author: wbs2788
Date: 2021-10-26 00:42:01
LastEditTime: 2021-10-30 00:02:16
LastEditors: wbs2788
Description: 
FilePath: \MCTS\policy_value_net.py

'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from game import Board

def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class Net(nn.Module):
    def __init__(self, board_width, board_height):
        super(Net, self).__init__()

        self.board_width = board_width
        self.board_height = board_height

        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.act_conv1 = nn.Conv2d(128, 4, kernel_size=4)
        self.act_fc1 = nn.Linear(4*board_width*board_height, board_width*board_height)    

        self.val_conv1 = nn.Conv2d(128, 2, kernel_size=1)
        self.val_fc1 = nn.Linear(2*board_width*board_height, 64)
        self.val_fc2 = nn.Linear(64, 1)

    def forward(self, state_input):

        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x_act = F.relu(self.act_conv1(x))
        x_act = x_act.view(-1, 4*self.board_width*self.board_height)
        x_act = F.log_softmax(self.act_fc1(x_act))

        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(-1, 2*self.board_width*self.board_height)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = F.tanh(self.val_fc2(x_val))

        return x_act, x_val

class PolicyValueNet():
    def __init__(self, board_width, board_height, model_file=None, use_gpu=False) -> None:
        self.use_gpu = use_gpu
        self.board_width = board_width
        self.board_height = board_height
        self.l2_const = 1e-4

        if self.use_gpu:
            self.policy_value_net = Net(board_width, board_height).cuda()
        else:
            self.policy_value_net = Net(board_width, board_height)

        self.optimizer = optim.Adam(self.policy_value_net.parameters(), weight_decay=self.l2_const)

        if model_file:
            net_params = torch.load(model_file)
            self.policy_value_net.load_state_dict(net_params)

    def policy_val(self, batch):
        if self.use_gpu:
            batch = Variable(torch.FloatTensor(batch).cuda())
            log_act_probs, val = self.policy_value_net(batch)
            act_probs = np.exp(log_act_probs.data.cpu().numpy)
            return act_probs, val.data.numpy()
        else:
            batch = Variable(torch.FloatTensor(batch))
            log_act_probs, val = self.policy_value_net(batch)
            act_probs = np.exp(log_act_probs.data.numpy)
            return act_probs, val.data.numpy()

    def policy_val_func(self, board:Board):
        legal_pos = board.available
        cur_states = np.ascontiguousarray(board.current_state().reshape(
            -1, 4, self.board_width, self.board_height))
        if self.use_gpu:
            log_act_probs, value = self.policy_value_net(
                Variable(torch.from_numpy(cur_states)).cuda().float())
            act_probs = np.exp(log_act_probs.data.cpu().numpy().flatten())
        else:
            log_act_probs, value = self.policy_value_net(
                    Variable(torch.from_numpy(cur_states)).float())
            act_probs = np.exp(log_act_probs.data.numpy().flatten())    
        act_probs = zip(legal_pos, act_probs[legal_pos])
        value = value.data[0][0]
        return act_probs, value

    def train_step(self, batch, mcts_probs, win_batch, lr):
        if self.use_gpu:
            batch = Variable(torch.FloatTensor(batch).cuda())
            mcts_probs = Variable(torch.FloatTensor(mcts_probs).cuda())
            win_batch = Variable(torch.FloatTensor(win_batch).cuda())
        else:
            batch = Variable(torch.FloatTensor(batch))
            mcts_probs = Variable(torch.FloatTensor(mcts_probs))
            win_batch = Variable(torch.FloatTensor(win_batch))

        self.optimizer.zero_grad()
        set_lr(self.optimizer, lr)

        log_act_probs, value = self.policy_value_net(batch)
        value_loss = F.mse_loss(value.view(-1), win_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs*log_act_probs, 1))
        loss = value_loss + policy_loss

        loss.backward()
        self.optimizer.step()
        entropy = -torch.mean(
                torch.sum(torch.exp(log_act_probs) * log_act_probs, 1))
        
        return loss.item(), entropy.item()

    def get_policy_param(self):
        return self.policy_value_net.state_dict()

    def save(self, model_file):
        net_params = self.get_policy_param()
        torch.save(net_params, model_file)