import tqdm
from torch.autograd import Variable
import numpy as np

from .base import Callback
from .inference import Inferencer
from ..utils import get_tqdm_kwargs
from ..dataflow import DataFlow

import pdb

__all__ = ['InferenceRunner']


class InferenceRunner(Callback):

    def __init__(self, ds, infs):
        self._ds = ds
        if not isinstance(infs, list):
            self.infs = [infs]
        else:
            self.infs = infs
        for v in self.infs:
            assert isinstance(v, Inferencer), v
        self.inf_summaries = { }

    def setup_graph(self, trainer):
        self.trainer = trainer
        self.model = self.trainer.model
        self._input_desc = self.model.get_inputs()

    def trigger(self):
        self._ds.reset_state()
        self.model.eval()
        for _ in tqdm.trange(self._ds.size(), **get_tqdm_kwargs(leave=True)):
            dp = next(self._ds.get_data())
            assert len(dp) == len(self._input_desc), "Length of inputs should be same with length of input_desc"
            dp = [Variable(self._input_desc[idx](ele).cuda()) for idx, ele in enumerate(dp)]
            self.model.run_graph(dp)
            for inf in self.infs:
                if inf.inf_name not in self.inf_summaries.keys():
                    self.inf_summaries[inf.inf_name] = []
                output = []
                for var_name in inf.var_names:
                    output.append(self.model.__dict__[var_name].data)
                summary_val = inf.datapoint(output)
                self.inf_summaries[inf.inf_name].extend(summary_val)

        for name, values in self.inf_summaries.items():
            self.trainer.monitors.put_scalar(name, np.mean(values))
        self.clear_summaries()

    def clear_summaries(self):
        self.inf_summaries = { }
