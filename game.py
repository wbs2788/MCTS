'''
Author: wbs2788
Date: 2021-10-19 00:33:35
LastEditTime: 2021-10-29 23:58:22
LastEditors: wbs2788
Description: create game
FilePath: \MCTS\game.py
'''

import numpy as np

class Board(object):

    def __init__(self, **kwargs):
        self.width = int(kwargs.get('width', 8))
        self.height = int(kwargs.get('height', 8))
        self.states = {} # key: loc. val: player
        self.n_in_row = int(kwargs.get('n_in_row', 5))
        self.players = [1, 2]

    def init_board(self, start=0):
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

    def loc2move(self, loc):
        if len(loc) != 2 :
            return -1
        if loc[0] >= self.height or loc[1] >= self.width:
            return -1
        else:
            return loc[0] * self.width + loc[1] 

    def current_state(self):

        states4 = np.zeros((4, self.width, self.height))
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            cur_move = moves[players == self.cur_player]
            opp_move = moves[players != self.cur_player]
            states4[0][cur_move // self.width, cur_move % self.height] = 1.0
            states4[1][opp_move // self.width, opp_move % self.height] = 1.0
            states4[2][self.last_move // self.width, self.last_move % self.height]  = 1.0
            if len(self.states) % 2 == 0:
                states4[3][:,:] = 1.0
            return states4[:, ::-1, :]

    def do_move(self, move):
        self.states[move] = self.cur_player
        self.available.remove(move)
        if self.cur_player == self.players[0]:
            self.cur_player = self.players[1]
        else:
            self.cur_player = self.players[0]

    def whoswin(self):
        w = self.width
        h = self.height
        states = self.states
        n = self.n_in_row

        moved = list(set(range(w * h)) - set(self.available))
        if len(moved) < 2*n - 1:
            return False, -1
        
        for m in moved:
            player = states[m]
            if (m // w in range(w - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n))) == 1):
                return True, player

            if (m % h in range(h - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * w, w))) == 1):
                return True, player

            if (m // w in range(w - n + 1) and m % h in range(h - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (w + 1), w + 1))) == 1):
                return True, player

            if (m // w in range(n - 1, w) and m % h in range(h - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (w - 1), w - 1))) == 1):
                return True, player

        return False, -1

    def checkend(self):
        win, winner = self.whoswin()
        if win:
            return True, winner
        elif not len(self.available):
            return True, -1
        return False, -1

    def  get_cur_player(self):
        return self.cur_player

        
class Game(object):
 
    def __init__(self, board, **kwargs):
        self.board = board

    def graphic(self, board, player1, player2):
        w = board.width
        h = board.height

        print("Player", player1, "with X".rjust(3))
        print("Player", player2, "with O".rjust(3))
        print()
        for x in range(w):
            print("{0:8}".format(x), end='')
        print('\r\n')
        for i in range(t - 1, -1, -1):
            print("{0:4d}".format(i), end='')
            for j in range(w):
                loc = i * w + j
                p = board.states.get(loc, -1)
                if p == player1:
                    print('X'.center(8), end='')
                elif p == player2:
                    print('O'.center(8), end='')
                else:
                    print('_'.center(8), end='')
            print('\r\n\r\n')

    def start_play(self, player1, player2, start_player=0, is_shown=1):

        if start_player not in (0, 1):
            raise Exception('start_player should be either 0 (player1 first) '
                            'or 1 (player2 first)')
        self.board.init_board(start_player)
        p1, p2 = self.board.players
        player1.set_player_ind(p1)
        player2.set_player_ind(p2)
        players = {p1: player1, p2: player2}
        if is_shown:
            self.graphic(self.board, player1.player, player2.player)
        while True:
            current_player = self.board.get_current_player()
            player_in_turn = players[current_player]
            move = player_in_turn.get_action(self.board)
            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, player1.player, player2.player)
            end, winner = self.board.game_end()
            if end:
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is", players[winner])
                    else:
                        print("Game end. Tie")
                return winner

    def start_self_play(self, player, is_shown=0, temp=1e-3):
        self.board.init_board()
        p1, p2 = self.board.players
        states, mcts_probs, current_players = [], [], []
        while True:
            move, move_probs = player.get_action(self.board,
                                                 temp=temp,
                                                 return_prob=1)
            # store the data
            states.append(self.board.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player)
            # perform a move
            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, p1, p2)
            end, winner = self.board.game_end()
            if end:
                # winner from the perspective of the current player of each state
                winners_z = np.zeros(len(current_players))
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                # reset MCTS root node
                player.reset_player()
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is player:", winner)
                    else:
                        print("Game end. Tie")
                return winner, zip(states, mcts_probs, winners_z)