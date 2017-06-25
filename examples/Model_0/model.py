import argparse
import tushare as ts

import torch
import torch.nn as nn
from torch.autograd import Variable

from finpack import *

from reader import StockHistory
from cfgs.config import cfg

import pdb

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.n_layers, batch_first=True)
        self.decoder = nn.Linear(self.hidden_size, self.output_size)


    def forward(self, input):
        shape = input.size()
        batch_size = shape[0]
        seq_len = shape[1]

        hidden =  Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size).cuda())
        cell = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size).cuda())

        output, h_c = self.lstm(input, (hidden, cell))

        output = output.contiguous().view(batch_size * seq_len, self.hidden_size)
        output = self.decoder(output)
        output = output.view(batch_size, seq_len, self.output_size)
        return output, h_c

class Model():
    def __init__(self):
        self.module = RNN(len(cfg.predictors), cfg.hidden_size, cfg.output_size, cfg.n_layers)
        self.module.cuda()

    def get_inputs(self):
        return [torch.FloatTensor, torch.FloatTensor]

    def run_graph(self, inputs):
        [input, label] = inputs

        output, (hidden, cell) = self.module(input)
        self.cost = criterion(output[:,-1], label)

    def get_optimizer(self):
        return torch.optim.Adam(self.module.parameters(), lr=cfg.lr)

def get_config(args):
    stock_list = ts.get_hs300s()['code'].as_matrix().tolist()
    # stock_list = ["600000"]

    ds_train = StockHistory(stock_list,
                            start="2010-01-01",
                            end="2017-05-31",
                            pred_column="close")

    ds_train = BatchData(ds_train, int(args.batch_size))

    callbacks = []

    return TrainConfig(
        dataflow=ds_train,
        callbacks=callbacks,
        model=Model(),
        max_epoch=160,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', help='batch size', required=True)
    args = parser.parse_args()

    config = get_config(args)
    SimpleTrainer(config).train()
