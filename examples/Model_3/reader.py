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
        start_year = int(start.split('-')[0]) - 2
        month = start.split('-')[1]
        day = start.split('-')[2]
        self.start = str(start_year) + '-' + month + '-' + day
        self.label_start = start
        self.end = end
        self.pred_column = pred_column
        self.seq_len = seq_len
        self.shuffle = shuffle


        self.input_list = []
        self.label_list = []
        logger.info("Loading Data")
        for stock in tqdm(stock_list, ascii=True):
            # stock_data = ts.get_k_data(stock, start=start, end=end)
            stock_data = get_data_by_code(stock, self.start, end)
            temp = stock_data[stock_data['date'] >= self.label_start].index.tolist()
            if len(temp) == 0:
                continue
            label_idx = temp[0]
            start_idx = stock_data[0:1].index.tolist()[0]
            label_idx = label_idx - start_idx
            input_data = stock_data[cfg.predictors].as_matrix()
            label_data = stock_data[pred_column].as_matrix()

            length = input_data.shape[0]

            print("00000")
            start_idx = np.max([label_idx, seq_len])
            end_idx = length - 1
            print(stock_data[start_idx:start_idx+1])
            print(stock_data[end_idx:end_idx+1])

            predict_interval = 5

            for idx in range(np.max([label_idx, seq_len]), length - predict_interval + 1):
                cur_input_p = input_data[idx - seq_len: idx, 0:4]
                cur_input_p = np.transpose(cur_input_p, (1, 0))
                cur_input_v = input_data[idx - seq_len: idx, 4:]
                cur_input_v = np.transpose(cur_input_v, (1, 0))
                if label_data[idx + predict_interval - 1] > label_data[idx - 1 + predict_interval - 1]:
                    cur_label = 1
                else:
                    cur_label = 0

                cur_input_short_p = cur_input_p[:,-20:]
                cur_input_short_v = cur_input_v[:,-20:]

                group = 5
                cur_input_middel_p = np.zeros((4, 20))
                cur_input_middel_v = np.zeros((1, 20))
                middel_data_p = cur_input_p[:,-100:]
                middel_data_v = cur_input_v[:,-100:]
                for i in range(20):
                    open_v = middel_data_p[0, i * group]
                    close_v = middel_data_p[1, i * group + group - 1]
                    high_v = np.max(middel_data_p[2, i * group:i * group + group])
                    low_v = np.min(middel_data_p[3, i * group:i * group + group])
                    volume_v = np.sum(middel_data_v[:, i * group:i * group + group], axis=1)
                    cur_input_middel_p[:, i] = np.asarray([open_v, close_v, high_v, low_v])
                    cur_input_middel_v[:, i] = volume_v

                group = 12
                cur_input_long_p = np.zeros((4, 20))
                cur_input_long_v = np.zeros((1, 20))
                long_data_p = cur_input_p[:,-240:]
                long_data_v = cur_input_v[:,-240:]
                for i in range(20):
                    open_v = long_data_p[0, i * group]
                    close_v = long_data_p[1, i * group + group - 1]
                    high_v = np.max(long_data_p[2, i * group:i * group + group])
                    low_v = np.min(long_data_p[3, i * group:i * group + group])
                    volume_v = np.sum(long_data_v[:, i * group:i * group + group], axis=1)
                    cur_input_long_p[:, i] = np.asarray([open_v, close_v, high_v, low_v])
                    cur_input_long_v[:, i] = volume_v


                if cfg.normalize == True:
                    max_v = np.max(cur_input_short_p)
                    min_v = np.min(cur_input_short_p)
                    cur_input_short_p = (cur_input_short_p - min_v) / (max_v - min_v)
                    max_v = np.max(cur_input_middel_p)
                    min_v = np.min(cur_input_middel_p)
                    cur_input_middel_p = (cur_input_middel_p - min_v) / (max_v - min_v)
                    max_v = np.max(cur_input_long_p)
                    min_v = np.min(cur_input_long_p)
                    cur_input_long_p = (cur_input_long_p - min_v) / (max_v - min_v)

                    max_v = np.max(cur_input_short_v)
                    cur_input_short_v = cur_input_short_v / max_v
                    max_v = np.max(cur_input_middel_v)
                    cur_input_middel_v = cur_input_middel_v / max_v
                    max_v = np.max(cur_input_long_v)
                    cur_input_long_v = cur_input_long_v / max_v

                cur_input_short_p = np.expand_dims(cur_input_short_p, 0)
                cur_input_middel_p = np.expand_dims(cur_input_middel_p, 0)
                cur_input_long_p = np.expand_dims(cur_input_long_p, 0)
                cur_input_short_v = np.expand_dims(cur_input_short_v, 0)
                cur_input_middel_v = np.expand_dims(cur_input_middel_v, 0)
                cur_input_long_v = np.expand_dims(cur_input_long_v, 0)

                cur_label = np.expand_dims(cur_label, 0)

                self.input_list.append([cur_input_short_p,
                                        cur_input_middel_p,
                                        cur_input_long_p,
                                        cur_input_short_v,
                                        cur_input_middel_v,
                                        cur_input_long_v])
                self.label_list.append(cur_label)

    def size(self):
        return len(self.input_list)

    def get_data(self):
        idxs = np.arange(len(self.input_list))
        if self.shuffle == True:
            self.rng.shuffle(idxs)
        for k in idxs:
            yield [self.input_list[k][0],
                   self.input_list[k][1],
                   self.input_list[k][2],
                   self.input_list[k][3],
                   self.input_list[k][4],
                   self.input_list[k][5],
                   self.label_list[k]]

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


