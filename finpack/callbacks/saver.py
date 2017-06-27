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
        torch.save(self.trainer.model.get_saved_model(), self.path)

        # try:
        #     if not self.meta_graph_written:
        #         self.saver.export_meta_graph(
        #             os.path.join(self.checkpoint_dir,
        #                          'graph-{}.meta'.format(logger.get_time_str())),
        #             collection_list=self.graph.get_all_collection_keys())
        #         self.meta_graph_written = True
        #     self.saver.save(
        #         tf.get_default_session(),
        #         self.path,
        #         global_step=tf.train.get_global_step(),
        #         write_meta_graph=False)
        #     logger.info("Model saved to %s." % tf.train.get_checkpoint_state(self.checkpoint_dir).model_checkpoint_path)
        # except (OSError, IOError, tf.errors.PermissionDeniedError,
        #         tf.errors.ResourceExhaustedError):   # disk error sometimes.. just ignore it
        #     logger.exception("Exception in ModelSaver!")
