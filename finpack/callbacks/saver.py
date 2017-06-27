import torch
import os
import shutil
import glob

from .base import Callback
from ..utils import logger

__all__ = ['ModelSaver']


class ModelSaver(Callback):
    """
    Save the model every epoch.
    """

    def __init__(self, checkpoint_dir=None):
        """
        Args:
            checkpoint_dir (str): Defaults to ``logger.LOG_DIR``.
        """
        if checkpoint_dir is None:
            checkpoint_dir = logger.LOG_DIR
        assert os.path.isdir(checkpoint_dir), checkpoint_dir
        self.checkpoint_dir = checkpoint_dir

    def _setup_graph(self):
        self.path = os.path.join(self.checkpoint_dir, 'model')

    def _trigger(self):
        filepath = self.path + "-" + str(self.trainer.global_step)
        torch.save(self.trainer.model.get_saved_model().state_dict(), filepath)
