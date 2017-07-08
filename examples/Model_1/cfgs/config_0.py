import numpy as np
from easydict import EasyDict as edict

cfg = edict()

cfg.predictors = "open,close,high,low".split(',')

cfg.epoch = 200
# cfg.input_size = 1
cfg.hidden_size = 100
cfg.n_layers = 1
cfg.output_size = 1
cfg.lr = 1e-3

cfg.normalize = True
