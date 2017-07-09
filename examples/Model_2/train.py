import argparse
import tushare as ts

import torch
import torch.nn as nn
from torch.autograd import Variable

from finpack import *
from finpack.utils import logger

from reader import StockHistory
from cfgs.config import cfg

import pdb

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(4, 16, 5)
        self.conv2 = nn.Conv1d(16, 16, 5)
        # self.conv3 = nn.Conv1d(16, 32, 5)

        self.fc1 = nn.Linear(16 * 12, 32)
        self.fc2 = nn.Linear(32, 2)

        self.leaky_relu = nn.LeakyReLU(0.1)


    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.conv2(x))
        # x = self.leaky_relu(self.conv3(x))
        x = x.view(-1, 16 * 12)
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
        return [torch.FloatTensor, torch.LongTensor]

    def _run_graph(self, inputs):
        [input, label] = inputs

        self.label = label

        self.output = self.module(input)

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
    stock_list = ts.get_hs300s()['code'].as_matrix().tolist()
    # stock_list = ["600000"]

    ds_train = StockHistory(stock_list[:20],
                            start="2010-01-01",
                            end="2016-12-31",
                            pred_column="close",
                            seq_len=cfg.seq_len)

    ds_test = StockHistory(stock_list[:20],
                           start="2017-01-01",
                           end="2017-5-31",
                           pred_column="close",
                           seq_len=cfg.seq_len)

    augmentors = [
        augs.GaussianNoise(0.1)
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
