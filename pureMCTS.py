'''
Author: wbs2788
Date: 2021-10-29 10:12:19
LastEditTime: 2021-10-29 23:55:53
LastEditors: wbs2788
Description: file information
'''

import game
import copy
from operator import itemgetter

import numpy as np

def rollout_policy_func(board:game.Board):
    action_probs = np.random.rand(len(board.available))
    return zip(board.available, action_probs)

def policy_value_func(board:game.Board):
    action_probs = np.ones(len(board.available)) / len(board.available)
    return zip(board.available, action_probs), 0


class Node(object):
    def __init__(self, parent, priP:float) -> None:
        """initialize MCT

        Args:
            parent (Node): parent node
            priP (float): prior probability
        """    
        self._parent = parent
        self._children = {}
        self._cnt = 0
        self._q = 0
        self._u = 0
        self._p = priP

    def get_value(self, score:int):
        """Calculate value of a node

        Args:
            score ([type]): [description]

        Returns:
            [type]: [description]
        """        
        self._u = (score * self._p * np.sqrt(self._parent._cnt) / (1 + self._cnt))
        return self._u + self._q

    def select(self, score:int):
        """select

        Args:
            score ([type]): [description]

        Returns:
            [type]: [description]
        """        
        return max(self._children.items(), key=lambda node: node[1].get_value(score))
        
    def expand(self, action_prob):
        """expand

        Args:
            action_prob (list((action, prob))): list of cur node's action and prob
        """        
        for action, prob in action_prob:
            if action not in self._children:
                self._children[action] = Node(self, prob)

    def update(self, leaf_val):
        if self._parent:
            self._parent.update(-leaf_val)
        self._cnt += 1
        self._q += float((leaf_val - self._q)) / self._cnt

    def is_leaf(self):
        return self._children == {}

    def is_root(self):
        return self._parent is None

class MCTS(object):

    def __init__(self, policy_val_func, score=5, n_playout=10000) -> None:
        super().__init__()
        self._root = Node(None, 1.0)
        self._policy = policy_val_func
        self._score = score
        self._n_playout = n_playout

    def _playout(self, state:game.Board):
        node = self._root
        while True:
            if node.is_leaf():
                break
            action, node = node.select(self._score)
            state.do_move(action)

        action_probs, _ = self._policy(state)
        end, winner = state.checkend()    

        if not end:
            node.expand(action_probs)

        leaf_val = self._evaluate_rollout(state)
        node.update(-leaf_val)

    def _evaluate_rollout(self, state:game.Board, limit=1000):
        player = state.get_cur_player()
        for _ in range(limit):
            end, winner = state.checkend()
            if end:
                break
            action_probs = rollout_policy_func(state)
            max_action = max(action_probs, key=itemgetter(1))[0]
            state.do_move(max_action)
        else:
            print("WARNING: rollout reached move limit")
        if winner == -1:
            return 0
        else:
            return 1 if winner == player else -1

    def get_move(self, state):
        for _ in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)
        return max(self._root._children.items(),
                    key=lambda act_node: act_node[1]._n_visits)[0]

    def update_with_move(self, last_move):
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = Node(None, 1.0)

    def __str__(self):
        return "MCTS"

class MCTSPlayer(object):
    def __init__(self, score=5, n_playout=2000) -> None:
        super().__init__()
        self.mcts = MCTS(policy_value_func, score, n_playout)

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, board:game.Board):
        sensible_moves = board.available
        if len(sensible_moves) > 0:
            move = self.mcts.get_move(board)
            self.mcts.update_with_move(-1)
            return move
        else:
            print("WARNING: the board is FULL!")

    def __str__(self) -> str:
        return "MCTS {}".format(self.player)