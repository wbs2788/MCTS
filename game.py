'''
Author: wbs2788
Date: 2021-10-19 00:33:35
LastEditTime: 2021-10-19 00:49:45
LastEditors: wbs2788
Description: create game
FilePath: \MCTS\game.py
'''

import numpy as np

class Board(object):

    def __init__(self, **kwargs):
        self.width = int(kwargs.get('width', 8))
        self.height = int(kwargs.get('width', 8))
        self.states = {} # key: loc. val: player
        self.n_in_row = int(kwargs.get('n_in_row', 5))
        self.players = [1, 2]

    def init_Board(self, start=0):
        """initial the board

        Args:
            start (int): start player. Defaults to 0.

        Raises:
            Exception: Board is too small
        """        
        if self.width < self.n_in_row or self.height < self.n_in_row:
            raise Exception('Board is too small to make {} in a row!'.format(self.n_in_row))
        self.cur_player = self.players[start]
        self.available = list(range(self.width * self.height))
        self.states = {}
        self.last_move = -1

    def move_location(self, movement):
        """get movement index

        Args:
            movement (int): give a number of location

        Returns:
            [int, int]: give coordinate of movement
        """        
        h = movement // self.width
        w = movement % self.height
        return [h, w]

    