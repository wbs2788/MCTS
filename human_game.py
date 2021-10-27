'''
Author: wbs2788
Date: 2021-10-20 13:18:56
LastEditTime: 2021-10-27 23:23:20
LastEditors: wbs2788
Description: 
FilePath: \MCTS\human_game.py
'''

import pickle
from game import Board, Game
from MCTS import MCTSPlayer
from policy_value_net import PolicyValueNet

class Human(object):
    
    def __init__(self) -> None:
        super().__init__()
        self.player = None

    def set_player_id(self, p):
        self.player = p

    def get_action(self, board:Board):
        try:
            location = input("type where to locate(format: x,y):")
            if isinstance(location, str):
                location = [int(n, 10) for n in location.split(",")]
            move = board.loc2move(location)
        except Exception as _:
            move = -1
        if move not in board.available:
            print("INVALID MOVE!!!")
            move = self.get_action(board)
        return move
    
    def __str__(self) -> str:
        return "human {}".format(self.player)

def run():
    n = 5
    width, height = 8, 8
    