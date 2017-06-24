import tushare as ts
import os
import pickle
from abc import abstractmethod, ABCMeta
import six

import pdb

from torch.autograd import Variable
import torch

from finpack import *

data_dir = "data"
data_files = os.listdir(data_dir)
filename_dict = {}
for data_file in data_files:
    code = data_file.split('_')[0]
    filename_dict[code] = os.path.join(data_dir, data_file)

def get_data_by_code(code):
    filepath = filename_dict[code]
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data


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
            # stock_data = ts.get_k_data(stock, start=start, end=end)
            stock_data = get_data_by_code(stock)
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

if __name__ == "__main__":
    data_dir = "data"

    stock_list = ts.get_stock_basics()
    stock_list = stock_list.index.tolist()

    tot_count = 0

    for code in stock_list:
        print(code)
        cur_data = ts.get_k_data(code, start="2000-01-01", end="2017-05-31")
        count = cur_data.shape[0]
        tot_count += count
        with open(os.path.join(data_dir, code + "_" + str(count)), 'wb') as f:
            pickle.dump(cur_data, f)

    print("Number of data points: " + str(tot_count))


