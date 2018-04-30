import numpy as np
import torch


class Env(object):

    def __init__(self, train_df, train_label, init_act):
        self.init_act = init_act

        self.i_step = 0
        self.i_act = init_act
        self.OHLCV = None
        self.reward = None

        self.train_df = torch.from_numpy(np.expand_dims(train_df.values, axis=1).astype(np.float32))
        self.train_label = train_label.values
        self.iter = None

    def _pos_encode(self):
        pos = torch.zeros((1, 5))
        pos[0, self.i_act] = 1.
        return pos

    def reset(self):
        self.i_step = 0
        self.i_act = self.init_act
        self.OHLCV = None
        self.reward = None

        self.iter = zip(iter(self.train_df), iter(self.train_label))

        s0, self.OHLCV = next(self.iter)
        return torch.cat((s0, self._pos_encode()), dim=1)

    def step(self, a):
        self.i_step += 1
        s_, self.OHLCV = next(self.iter)

        # reward
        open = self.OHLCV[0]
        close = self.OHLCV[3]
        self.reward = a * (close - open) / open

        # swap action
        self.i_act = a

        # done
        done = False if self.i_step < len(self.train_df) - 1 else True
        return torch.cat((s_, self._pos_encode()), dim=1), self.reward, done

    def curr_OHLCV(self):
        return self.OHLCV
