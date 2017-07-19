import argparse
import tushare as ts

import torch
import torch.nn as nn
from torch.autograd import Variable

from finpack import *
from finpack.utils import logger

from reader import StockHistory
from cfgs.config import cfg

from dataio import dataio

import pdb

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1_short_p = nn.Conv1d(4, 16, 5)
        self.conv2_short_p = nn.Conv1d(16, 16, 5)
        self.conv3_short_p = nn.Conv1d(16, 32, 5)

        self.conv1_middel_p = nn.Conv1d(4, 16, 5)
        self.conv2_middel_p = nn.Conv1d(16, 16, 5)
        self.conv3_middel_p = nn.Conv1d(16, 32, 5)

        self.conv1_long_p = nn.Conv1d(4, 16, 5)
        self.conv2_long_p = nn.Conv1d(16, 16, 5)
        self.conv3_long_p = nn.Conv1d(16, 32, 5)

        self.conv1_short_v = nn.Conv1d(1, 8, 5)
        self.conv2_short_v = nn.Conv1d(8, 16, 5)
        self.conv3_short_v = nn.Conv1d(16, 32, 5)

        self.conv1_middel_v = nn.Conv1d(1, 8, 5)
        self.conv2_middel_v = nn.Conv1d(8, 16, 5)
        self.conv3_middel_v = nn.Conv1d(16, 32, 5)

        self.conv1_long_v = nn.Conv1d(1, 8, 5)
        self.conv2_long_v = nn.Conv1d(8, 16, 5)
        self.conv3_long_v = nn.Conv1d(16, 32, 5)

        self.fc1 = nn.Linear(32 * 8 * 3 * 2, 64)
        self.fc2 = nn.Linear(64, 2)

        self.leaky_relu = nn.LeakyReLU(0.1)


    def forward(self, x_short, x_middel, x_long, v_short, v_middel, v_long):
        x_short = self.leaky_relu(self.conv1_short_p(x_short))
        x_short = self.leaky_relu(self.conv2_short_p(x_short))
        x_short = self.leaky_relu(self.conv3_short_p(x_short))

        x_middel = self.leaky_relu(self.conv1_middel_p(x_middel))
        x_middel = self.leaky_relu(self.conv2_middel_p(x_middel))
        x_middel = self.leaky_relu(self.conv3_middel_p(x_middel))

        x_long = self.leaky_relu(self.conv1_long_p(x_long))
        x_long = self.leaky_relu(self.conv2_long_p(x_long))
        x_long = self.leaky_relu(self.conv3_long_p(x_long))


        v_short = self.leaky_relu(self.conv1_short_v(v_short))
        v_short = self.leaky_relu(self.conv2_short_v(v_short))
        v_short = self.leaky_relu(self.conv3_short_v(v_short))

        v_middel = self.leaky_relu(self.conv1_middel_v(v_middel))
        v_middel = self.leaky_relu(self.conv2_middel_v(v_middel))
        v_middel = self.leaky_relu(self.conv3_middel_v(v_middel))

        v_long = self.leaky_relu(self.conv1_long_v(v_long))
        v_long = self.leaky_relu(self.conv2_long_v(v_long))
        v_long = self.leaky_relu(self.conv3_long_v(v_long))

        x_short = x_short.view(-1, 32 * 8)
        x_middel = x_middel.view(-1, 32 * 8)
        x_long = x_long.view(-1, 32 * 8)

        v_short = v_short.view(-1, 32 * 8)
        v_middel = v_middel.view(-1, 32 * 8)
        v_long = v_long.view(-1, 32 * 8)

        x = torch.cat((x_short, x_middel, x_long, v_short, v_middel, v_long), 1)

        x = self.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Model(ModelDesc):
    def __init__(self):
        super(Model, self).__init__()
        self.module = Net()
        self.module.cuda()
        self.criterion = nn.CrossEntropyLoss()
        self.criterion.cuda()

    def get_inputs(self):
        return [torch.FloatTensor,
                torch.FloatTensor,
                torch.FloatTensor,
                torch.FloatTensor,
                torch.FloatTensor,
                torch.FloatTensor,
                torch.LongTensor]

    def _run_graph(self, inputs):
        [input_short_p, input_middel_p, input_long_p, input_short_v, input_middel_v, input_long_v, label] = inputs

        self.label = label

        self.output = self.module(input_short_p, input_middel_p, input_long_p, input_short_v, input_middel_v, input_long_v)

        self.cost = self.criterion(self.output, self.label)

        self.add_summary(self.cost, "cost")

    def _get_optimizer(self):
        return torch.optim.SGD(self.module.parameters(), lr=cfg.lr, momentum=0.9)

    def get_saved_model(self):
        return self.module

    def _train(self):
        self.module.train()

    def _eval(self):
        self.module.eval()

def get_config(args):
    # stock_list = ts.get_hs300s()['code'].as_matrix().tolist()
    stock_list = list(dataio.get_classified(['金融行业']).keys())
    hs300 = list(dataio.get_hs300().keys())

    stock_list = list(set(stock_list).intersection(hs300))

    ds_train = StockHistory(stock_list,
                            start="2009-01-04",
                            end="2015-12-31",
                            pred_column="close",
                            seq_len=cfg.seq_len)

    ds_test = StockHistory(stock_list,
                           start="2016-01-01",
                           end="2016-5-31",
                           pred_column="close",
                           seq_len=cfg.seq_len)

    augmentors = [
        augs.GaussianNoise(0.01)
    ]

    ds_train = AugmentData(ds_train, augmentors)

    ds_train = BatchData(ds_train, int(args.batch_size))
    ds_test = BatchData(ds_test, int(args.batch_size))

    callbacks = [
        PeriodicTrigger(ModelSaver(), every_k_epochs=3),
        ScheduledHyperParamSetter('learning_rate', cfg.lr_sched),
        # InferenceRunner(ds_test, NumericError("cost")),
        InferenceRunner(ds_test, ClassificationError(["output", "label"], "val-classification-error")),
        LearningRateSetter(),
    ]

    return TrainConfig(
        dataflow=ds_train,
        callbacks=callbacks,
        model=Model(),
        max_epoch=cfg.epoch,
        load_path=args.load,
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', help='batch size', default=1024)
    parser.add_argument('--load', help='model to load')
    args = parser.parse_args()

    logger.auto_set_dir()
    config = get_config(args)
    SimpleTrainer(config).train()
