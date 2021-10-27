'''
Author: wbs2788
Date: 2021-10-27 23:23:43
LastEditTime: 2021-10-28 01:13:29
LastEditors: wbs2788
Description: 
FilePath: \MCTS\train.py

'''

import random
import numpy as np
from collections import defaultdict, deque
from game import Board, Game
from MCTS import MCTSPlayer
from policy_value_net import PolicyValueNet

class Train():

    def __init__(self, init_model=None) -> None:
        self.board_width = 8
        self.board_height = 8
        self.n_in_row = 5
        self.board = Board(width=self.board_width,
                           height=self.board_height, 
                           n_in_row=self.n_in_row)
        self.game = Game(self.board)

        self.lr = 1e-3
        self.lr_multiplier = 1.0
        self.temp = 1.0
        self.n_play = 1000
        self.score = 5
        self.buffer_size = 10000
        self.batchsize = 512
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        self.epochs = 10
        self.kl_targ = 0.02
        self.check_freq = 5
        self.game_batch_num = 1500
        self.best_win_ratio = 0.0

        if init_model:
            self.policy_val_net = PolicyValueNet(self.board_width,
                                                 self.board_height,
                                                 model_file=init_model)
        else:
            self.policy_val_net = PolicyValueNet(self.board_width,
                                                 self.board_height)

        self.mcts_player = MCTSPlayer(self.policy_val_net.policy_val_func,
                                      score=self.score,
                                      plays=self.n_playout,
                                      is_selfplay=1)



        