'''
Author: wbs2788
Date: 2021-10-27 23:23:43
LastEditTime: 2021-10-29 10:02:19
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

    def get_equi_data(self, play_data):
        extend_data = []
        for state, mcts_prob, winner in play_data:
            for i in [1,2,3,4]:
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(np.flipud(
                        mcts_prob.reshape(self.board_height, self.board_width)), i)
                extend_data.append((equi_state, np.flipud(equi_mcts_prob).flatten(), winner))
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state, np.flipud(equi_mcts_prob).flatten(), winner))
        return extend_data

    def collect_selfplay_data(self, n_games=1):
        for i in range(n_games):
            _, play_data = self.game.start_self_play(self.mcts_player, temp=self.temp)
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            play_data = self.get_equi_data(play_data)
            self.data_buffer.extend(play_data)

    def policy_update(self):
        minibatch = random.sample(self.data_buffer, self.batchsize)
        state_batch = [data[0] for data in minibatch]
        mcts_probs_batch = [data[1] for data in minibatch]
        winner_batch = [data[2] for data in minibatch]
        old_probs, old_v = self.policy_val_net.policy_val(state_batch)
        for _ in range(self.epochs):
            loss, entropy = self.policy_val_net.train_step(
                            state_batch, mcts_probs_batch, winner_batch,
                            self.lr*self.lr_multiplier)
            new_probs, new_v = self.policy_val_net.policy_val(state_batch)
            kl = np.mean(np.sum(old_probs*(np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
            axis=1))
            if kl > self.kl_targ*4:
                break
        if kl > self.kl_targ*2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 10

        explained_var_old = (1 -
                             np.var(np.array(winner_batch) - old_v.flatten()) /
                             np.var(np.array(winner_batch)))
        explained_var_new = (1 -
                             np.var(np.array(winner_batch) - new_v.flatten()) /
                             np.var(np.array(winner_batch)))
        
        print(("kl:{:.5f},"
               "lr_multiplier:{:.3f},"
               "loss:{},"
               "entropy:{},"
               "explained_var_old:{:.3f},"
               "explained_var_new:{:.3f}"
               ).format(kl,
                        self.lr_multiplier,
                        loss,
                        entropy,
                        explained_var_old,
                        explained_var_new))
        return loss, entropy
        
    def policy_evaluate(self, n_games=10):
        current_mcts_player = MCTSPlayer(self.policy_val_net.policy_val_func,
                                        score=self.score, plays=self.n_play)
        win_cnt = defaultdict(int)
        for i in range(n_games):
            winner = self.game.start_play(current_mcts_player)
            # !TODO: write pure MCTS
            win_cnt[winner] += 1
        win_ratio = 1.0*(win_cnt[1] + 0.5*win_cnt[-1]) / n_games
        return win_ratio

    def run(self):
        try:
            for i in range(self.game_batch_num):
                self.collect_selfplay_data(self.play_batch_size)
                print("batch i:{}, episode_len:{}".format(
                            i + 1, self.episode_len))
                if len(self.data_buffer) > self.batchsize:
                    loss, entropy = self.policy_update()
                if (i + 1) % self.check_freq == 0:
                    print("current self-play batch: {}".format(i+1))
                    win_ratio = self.policy_evaluate()
                    self.policy_val_net.save('./current_policy.model')
                    if win_ratio > self.best_win_ratio:
                        print("New best policy!!!!!!!!")
                        self.best_win_ratio = win_ratio
                        # update the best_policy
                        self.policy_val_net.save('./best_policy.model')
                        if (self.best_win_ratio == 1.0 and
                                self.pure_mcts_playout_num < 5000):
                            self.pure_mcts_playout_num += 1000
                            self.best_win_ratio = 0.0
        except KeyboardInterrupt:
            print('\n\rquit')

if __name__ == '__main__':
    training = Train()
    training.run()