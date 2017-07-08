import numpy as np
from easydict import EasyDict as edict

cfg = edict()

cfg.predictors = "open,close,high,low".split(',')

cfg.epoch = 200
# cfg.input_size = 1
cfg.hidden_size = 500
cfg.n_layers = 2
cfg.output_size = 2
cfg.lr = 1e-3

cfg.normalize = False
