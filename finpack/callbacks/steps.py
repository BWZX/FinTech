import tqdm

from ..utils import logger, get_tqdm_kwargs
from .base import Callback

__all__ = ['ProgressBar']

class ProgressBar(Callback):
    """ A progress bar based on tqdm. Enabled by default. """

    def __init__(self, names=[]):
        """
        Args:
            names(list): list of string, the names of the tensors to monitor
                on the progress bar.
        """
        super(ProgressBar, self).__init__()
        self._bar = None

    def _before_train(self):
        self._last_updated = self.local_step

        self._total = self.trainer.config.steps_per_epoch
        self._tqdm_args = get_tqdm_kwargs(leave=True)

    def _before_epoch(self):
        if self.local_step == 0:
            self._bar = tqdm.trange(self._total, **self._tqdm_args)

    def _trigger_step(self):
        self._bar.update()
        if self.local_step == self._total - 1:
            self._bar.close()

    def _after_train(self):
        if self._bar:       # training may get killed before the first step
            self._bar.close()
