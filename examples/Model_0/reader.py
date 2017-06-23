import tushare as ts
from abc import abstractmethod, ABCMeta
import six

from torch.autograd import Variable
import torch

from finpack import *

class StockHistory(DataFlow):
    def __init__(self, stock_list, start, end, input_columns=["close"], pred_column="close", seq_len=25, batch_size=1):
        if isinstance(stock_list, list) == False:
            stock_list = [stock_list]
        self.stock_list = stock_list
        self.start = start
        self.end = end
        if isinstance(input_columns, list) == False:
            input_columns = input_columns.split(',')
        self.input_columns = input_columns
        self.pred_column = pred_column
        self.seq_len=seq_len
        self.batch_size=batch_size

        # get data and save with pickle
        self.input_list = []
        self.label_list = []
        for stock in stock_list:
            stock_data = ts.get_k_data(stock, start=start, end=end)
            input_data = stock_data[input_columns].as_matrix()
            label_data = stock_data[pred_column].as_matrix()
            self.input_list.append(input_data)
            self.label_list.append(label_data)


    def get_data(self):

        for idx, input_seq in enumerate(self.input_list):
            reset = True
            idx = 0
            label_seq = self.label_list[idx]
            while True:
                if (idx + self.seq_len + 1 >= input_seq.shape[0]):
                    break
                input = Variable(torch.FloatTensor(input_seq[idx: idx + self.seq_len]).cuda())
                label = Variable(torch.FloatTensor(label_seq[idx + 1: idx + self.seq_len + 1]).cuda())
                input = input.unsqueeze(1)
                label = label.unsqueeze(1)
                yield ([input, label], reset)
                idx += self.seq_len
                reset = False

