import numpy as np
from easydict import EasyDict as edict

cfg = edict()

cfg.predictors = "open,close,high,low,volume".split(',')

cfg.epoch = 2000
# cfg.input_size = 1
cfg.lr = 1e-3
cfg.lr_sched = [(0, 1e-1), (400, 1e-2)]
cfg.seq_len = 240

cfg.normalize = True
