import matplotlib.pyplot as plt
import numpy as np
import os.path
import pickle
script_path = os.path.dirname(__file__)
plt.style.use('seaborn-paper')

from torch.autograd import Variable
import torch

from agent import Actor
from env import Env


def plot_result(title, r_lst, p_lst):
    """
    This function is EXTREMELY tedious and messy since I just want things to work and don't care that much.
    """
    plt.subplot(211)
    plt.title(title)

    # baseline1: BnH
    BnH = np.cumsum(np.insert(np.diff(np.log(np.array(p_lst))), 0, 0.)) * 2
    plt.plot(BnH, label='BnH')

    # baseline2: momentum
    import talib as ta
    np_p = np.array(p_lst)
    sma = ta.SMA(np_p, timeperiod=30)
    sma[np.isnan(sma)] = 0.

    # RL performance
    ret = np.append(np.diff(np.log(np_p)), 0.) * 4
    signal = np_p > sma
    mm_ret = np.cumsum(ret * signal)
    plt.plot(mm_ret, label='momentum')

    np_R = np.array(r_lst)
    log_R = np.log(np_R + 1)
    cum_R = np.cumsum(log_R)

    plt.xticks(())
    plt.ylabel('Cum. Log Returns')
    plt.plot(cum_R, label='RL')
    plt.legend()

    def MDD(x):
        _max = None
        temp = []
        for t in x:
            if _max is None or t > _max: _max = t
            temp.append(t - _max)
        return temp

    plt.subplot(212)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.xlabel('Step')
    plt.ylabel('MDD')
    plt.plot(MDD(BnH), label='BnH')
    plt.plot(MDD(mm_ret), label='momentum')
    plt.plot(MDD(cum_R), label='RL')

    plt.show()


def roll_out(env, model, train_mode):
    model.eval()

    ret = 0      # episode return
    r_lst = []   # store reward
    p_lst = []   # store price
    P_lst = []   # store position
    buffer = []

    s = env.reset()

    done = False
    while not done:

        # sample action
        A_Pr = model.forward(Variable(s))
        act = torch.multinomial(A_Pr.data, num_samples=1)
        i_act = act[0][0]

        # apply action
        s_, r, done = env.step(i_act)

        # tracker
        ret += r
        r_lst.append(r)
        p_lst.append(env.curr_OHLCV()[3])
        P_lst.append(i_act)

        # Save transitions
        buffer.append((s, act, r))

        if done: break

        # Swap states
        s = s_

    # Learning when episode finishes
    model.train()

    S, A, R = zip(*buffer)
    del buffer[:]

    S = Variable(torch.cat(S))
    A = Variable(torch.cat(A))

    # Compute target
    Q = []
    ret = 0
    for r in reversed(R):
        ret = r + .9 * ret
        Q.append(ret)
    Q.reverse()

    # standardize Q
    Q = np.array(Q).astype(np.float32)
    Q -= Q.mean()
    Q /= Q.std() + 1e-6
    Q.clip(min=-10, max=10)
    Q = np.expand_dims(Q, axis=1)

    Q = Variable(torch.from_numpy(Q))

    # PG update
    A_Pr = model.forward(S).gather(1, A).clamp(min=1e-7, max=1 - 1e-7)

    loss = -(Q * torch.log(A_Pr)).mean()
    model.optim.zero_grad()
    loss.backward()
    model.optim.step()

    model.eval()
    return ret, r_lst, p_lst, P_lst


def main(epochs):

    # create model
    model = Actor()

    # load data
    data_path = os.path.join(script_path, 'BTCUSD-15Min-Data.pkl')
    with open(data_path, 'rb') as handler:
        data_dict = pickle.load(handler)
    data = data_dict['data']
    label = data_dict['label']

    # train-test split 70% and 30%
    split = int(.7 * len(data))

    train_data = data.ix[:split]
    train_label = label.ix[:split]
    test_data = data.ix[split:]
    test_label = label.ix[split:]

    # train
    ret_lst = []
    for i_ep in range(epochs):
        idx = np.random.randint(len(train_data) - 96)

        # sample a consecutive of 96 steps
        env = Env(train_data[idx: idx + 96],
                  train_label[idx: idx + 96],
                  init_act=np.random.randint(5))
        ret = roll_out(env=env, model=model, train_mode=True)
        ret_lst.append(ret)

    # test
    env = Env(test_data,
              test_label,
              init_act=0)
    _, r_lst, p_lst, P_lst = roll_out(env=env, model=model, train_mode=False)

    # plot
    plot_result('Bitcoin', r_lst, p_lst)


if __name__ == '__main__':
    main(epochs=5000)
