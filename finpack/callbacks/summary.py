import numpy as np
import tensorflow as tf

from ..utils import logger
from .base import Callback

__all__ = ['MergeAllSummaries']


class MergeAllSummaries(Callback):

    def _trigger_epoch(self):
        self._trigger()

    def _trigger(self):
        for name, values in self.trainer.model.summaries.items():
            self.trainer.monitors.put_scalar(name, np.mean(values))
        self.trainer.model.clear_summaries()
