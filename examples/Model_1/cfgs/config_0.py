import numpy as np
from easydict import EasyDict as edict

cfg = edict()

cfg.predictors = "open,close,high,low".split(',')

cfg.epoch = 2000
# cfg.input_size = 1
cfg.hidden_size = 500
cfg.n_layers = 1
cfg.output_size = 2
cfg.lr = 1e-3
cfg.seq_len = 50

cfg.normalize = True
