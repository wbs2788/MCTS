'''
Author: wbs2788
Date: 2021-10-18 23:21:29
LastEditTime: 2021-10-19 00:31:54
LastEditors: wbs2788
Description: MCTS algorithm with AlphaGo style(policy-value network)
FilePath: \MCTS\MCTS.py
'''

import numpy as np

class Node(object):
    
    def __init__(self, parent, priP):
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

    def get_value(self, score):
        """Calculate value of a node

        Args:
            score ([type]): [description]

        Returns:
            [type]: [description]
        """        
        self._u = (score * self._p * np.sqrt(self._parent._cnt) / (1 + self._cnt))
        return self._u + self._q

    def select(self, score):
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
    
    def __init__(self, policy_val_func, score=5, plays=10000):
        """init

        Args:
            policy_val_func (func): a func to evaluate a list(action, prob) from a game and give a score
            score (int, optional): control the exploration depth from last moves. Defaults to 5.
            plays (int, optional): play times. Defaults to 10000.
        """        
        self._root = Node(None, 1.0)
        self._policy = policy_val_func
        self._score = score
        self._plays = plays

    def _singleplay(self, state):
        cur_node = self._root
        while not cur_node.is_leaf():
            cur_action, cur_node = cur_node.select(self._score)
            state.move(cur_action)

        action_prob, leaf_val = self._policy(state)
        end, winner = state.ifend() # check whether to end

        if not end:
            cur_node.expand(action_prob)
        else:
            if winner == -1: # tie
                leaf_val = 0.0
            elif winner == state.get_cur_player():
                leaf_val = 1.0
            else:
                leaf_val = -1.0
    
        cur_node.update(leaf_val)