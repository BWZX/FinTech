import torch
import numpy as np
from torch import nn
from torch.autograd import Variable
import tushare as ts

from reader import StockHistory
from model import RNN
from cfgs.config import cfg

def train_epoch():

    loss_ary = []

    for dp, reset in ds.get_data():
        # one step
        [input, label] = dp
        if reset is True:
            hidden, cell = rnn.init_hidden_cell()

        optimizer.zero_grad()

        output, (hidden, cell) = rnn(input, hidden, cell)
        loss = criterion(output, label)

        loss_ary.append(loss.data[0])

        hidden = Variable(hidden.data.cuda())
        cell = Variable(cell.data.cuda())

        loss.backward()
        optimizer.step()

    return np.mean(loss_ary)


if __name__ == '__main__':


    columns = "open,close,high,low".split(',')

    rnn = RNN(len(columns), cfg.hidden_size, cfg.output_size, cfg.n_layers)
    rnn.cuda()

    optimizer = torch.optim.Adam(rnn.parameters(), lr=cfg.lr)
    criterion = nn.MSELoss()

    stock_list = ts.get_hs300s()['code'].as_matrix().tolist()

    ds = StockHistory(stock_list,
                      start="2010-01-01",
                      end="2017-05-31",
                      input_columns=columns,
                      pred_column="close")

    loss_ary = []
    for epoch in range(1, cfg.epoch + 1):

        ds.reset_state()
        loss = train_epoch()
        loss_ary.append(loss)

        print("Epoch %d finished. Loss is %.2f" % (epoch, loss))
