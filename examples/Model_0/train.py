import torch
from termcolor import colored
import numpy as np
from tqdm import tqdm
from torch import nn
from torch.autograd import Variable
import tushare as ts

from reader import StockHistory
from model import RNN
from cfgs.config import cfg

from finpack.utils import logger
from finpack import *

import pdb

def train_epoch():

    ds.reset_state()

    loss_ary = []
    for dp in tqdm(ds.get_data(), total=ds.size(), ascii=True):

        # one step
        [input, label] = dp

        input = Variable(torch.FloatTensor(input).cuda())
        label = Variable(torch.FloatTensor(label).cuda())

        optimizer.zero_grad()

        output, (hidden, cell) = rnn(input)

        loss = criterion(output[:,-1], label)

        loss_ary.append(loss.data[0])

        loss.backward()
        optimizer.step()

    return np.mean(loss_ary)


if __name__ == '__main__':


    columns = "open,close,high,low".split(',')

    rnn = RNN(len(columns), cfg.hidden_size, cfg.output_size, cfg.n_layers)

    logger.info(colored("Model Parameters:", 'cyan'))
    logger.info(rnn)


    rnn.cuda()

    optimizer = torch.optim.Adam(rnn.parameters(), lr=cfg.lr)
    criterion = nn.MSELoss()

    stock_list = ts.get_hs300s()['code'].as_matrix().tolist()
    # stock_list = ["600000"]

    ds = StockHistory(stock_list,
                      start="2010-01-01",
                      end="2017-05-31",
                      input_columns=columns,
                      pred_column="close")

    ds = BatchData(ds, 1024, stack_dim=0)

    loss_ary = []
    for epoch in range(1, cfg.epoch + 1):

        ds.reset_state()
        logger.info("Start Epoch {}".format(epoch))
        loss = train_epoch()
        loss_ary.append(loss)

        print("Epoch %d finished. Loss is %.2f" % (epoch, loss))
