'''
Author: wbs2788
Date: 2021-10-29 10:12:19
LastEditTime: 2021-10-29 10:28:42
LastEditors: wbs2788
Description: file information
'''
from _typeshed import Self
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
    # TODO: copy the currect node
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

    def _playout(self, state:Node):
        node = self._root
        while True:
            if node.is_leaf():
                break
            action, node = node.select(self._score)
            state.move(action) # TODO: find state.