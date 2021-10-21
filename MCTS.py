'''
Author: wbs2788
Date: 2021-10-08 23:21:29
LastEditTime: 2021-10-20 23:05:46
LastEditors: wbs2788
Description: MCTS algorithm with AlphaGo style(policy-value network)
FilePath: \MCTS\MCTS.py
'''

import numpy as np
import copy

class Node(object):
    
    def __init__(self, parent, priP) -> None:
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
    
    def __init__(self, policy_val_func, score=5, plays=10000) -> None:
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

    def _singleplay(self, state) -> None:
        """run a single play from the root to the leaf.

        Args:
            state ([type]): [description]
        """        
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

    def get_mov_probs(self, state, eps=1e-3):
        for _ in range(self._plays):
            state_copy = copy.deepcopy(state)
            self._singleplay(state_copy)

        act_visits = [(act, node.plays) for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)
        x = 1.0/eps * np.log(np.array(visits) + 1e-10)    
        act_probs = np.exp(x - np.max(x))
        act_probs /= np.sum(act_probs)
        return act_probs

    def update_with_move(self, last_move) -> None:
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = Node(None, 1.0)
            
class MCTSPlayer(object):

    def __init__(self, policy_val_func, score=5, plays=2000, is_selfplay=0) -> None:
        self.mcts = MCTS(policy_val_func, score, plays)
        self._is_selfplay = is_selfplay

    def set_player_id(self, p):
        self.player = p
    
    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, board, eps=1e-3, return_prob=0):
        sensible_moves = board.availables
        move_probs = self.mcts.get_mov_probs(board, eps)
        if len(sensible_moves) > 0:
            acts, probs = self.mcts.get_mov_probs(board, eps)
            move_probs[list(acts)] = probs
            if self._is_selfplay:
                move = np.random.choice(acts, p=0.75*probs + 0.25*np.random.dirichlet(0.3*np.ones(len(probs))))
                self.mcts.update_with_move(move)
            else:
                move = np.random.choice(acts, p=probs)
                self.mcts.update_with_move(-1)
            if return_prob:
                return move, move_probs
            else:
                return move
        else:
            print("WARNING: Board is FULL!")

    def __str__(self) -> str:
        return "MCTS {}".format(self.player)