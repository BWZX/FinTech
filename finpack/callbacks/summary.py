import numpy as np
import tensorflow as tf

from ..utils import logger
from .base import Callback

__all__ = ['MergeAllSummaries']


class MergeAllSummaries(Callback):

    def _setup_graph(self):
        pass

    def _trigger_step(self):
        if (self.local_step + 1) % self._period == 0:
            self._trigger()

    def _trigger(self):
        for name, values in self.trainer.model.summaries:
            self.trainer.monitors.put_scalar(name, np.mean(values))
        self.trainer.model.clear_summaries()
