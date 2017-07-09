import tushare as ts
import numpy as np
from tqdm import tqdm
import os
import pickle
from abc import abstractmethod, ABCMeta
import six

import pdb

from torch.autograd import Variable
import torch

from finpack import *
from finpack.utils import logger

from cfgs.config import cfg

data_dir = "data"
data_files = os.listdir(data_dir)
filename_dict = {}
for data_file in data_files:
    code = data_file.split('_')[0]
    filename_dict[code] = os.path.join(data_dir, data_file)

def get_data_by_code(code, start, end):
    filepath = filename_dict[code]
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    start = data['date'] >= start
    end = data['date'] <= end
    data = data[start & end]
    return data


class StockHistory(RNGDataFlow):
    def __init__(self, stock_list, start, end, pred_column="close", seq_len=25, shuffle=True):
        if isinstance(stock_list, list) == False:
            stock_list = [stock_list]
        self.stock_list = stock_list
        self.start = start
        self.end = end
        self.pred_column = pred_column
        self.seq_len = seq_len
        self.shuffle = shuffle

        self.input_list = []
        self.label_list = []
        logger.info("Loading Data")
        for stock in tqdm(stock_list, ascii=True):
            # stock_data = ts.get_k_data(stock, start=start, end=end)
            stock_data = get_data_by_code(stock, start, end)
            input_data = stock_data[cfg.predictors].as_matrix()
            label_data = stock_data[pred_column].as_matrix()

            length = input_data.shape[0]

            for step in range(length - seq_len - 1):
                cur_input = input_data[step: step + seq_len]
                cur_input = np.transpose(cur_input, (1, 0))
                if label_data[step + seq_len] > label_data[step + seq_len - 1]:
                    cur_label = 1
                else:
                    cur_label = 0

                if cfg.normalize == True:
                    max_v = np.max(cur_input)
                    min_v = np.min(cur_input)
                    cur_input = (cur_input - min_v) / (max_v - min_v)

                cur_input = np.expand_dims(cur_input, 0)
                cur_label = np.expand_dims(cur_label, 0)

                self.input_list.append(cur_input)
                self.label_list.append(cur_label)

                # if len(self.input_list) >= 500:
                #     break

    def size(self):
        return len(self.input_list)

    def get_data(self):
        idxs = np.arange(len(self.input_list))
        if self.shuffle == True:
            self.rng.shuffle(idxs)
        for k in idxs:
            yield [self.input_list[k], self.label_list[k]]

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


