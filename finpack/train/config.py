from ..callbacks import (Callbacks, ProgressBar)
from ..dataflow.base import DataFlow
# from ..models import ModelDesc
from ..utils import logger

__all__ = ['TrainConfig']


class TrainConfig(object):
    """
    Config for trainer.
    """

    def __init__(self, dataflow, model=None, callbacks=None,
                 starting_epoch=1, steps_per_epoch=None, max_epoch=99999):
        """
        Args:
            dataflow (DataFlow): the dataflow to train.
            model (ModelDesc): the model to train.
            callbacks (list): a list of :class:`Callback` to perform during training.
            starting_epoch (int): The index of the first epoch.
            steps_per_epoch (int): the number of steps (defined by :meth:`Trainer.run_step`) to run in each epoch.
                Defaults to the input data size.
            max_epoch (int): maximum number of epoch to run training.
        """

        def assert_type(v, tp):
            assert isinstance(v, tp), v.__class__

        # process data
        self.dataflow = dataflow
        assert_type(self.dataflow, DataFlow)

        if callbacks is None:
            callbacks = []
        assert_type(callbacks, list)
        extra_callbacks = [
            ProgressBar()]
            # MergeAllSummaries(),
            # RunUpdateOps()]
        self._callbacks = callbacks + extra_callbacks
        assert_type(self._callbacks, list)

        # self.monitors = [SummaryWriter(), JSONWriter(), ScalarPrinter()]

        self.model = model
        # assert_type(self.model, ModelDesc)

        self.steps_per_epoch = self.dataflow.size()

        self.starting_epoch = int(starting_epoch)
        self.max_epoch = int(max_epoch)
        assert self.steps_per_epoch >= 0 and self.max_epoch > 0

    @property
    def callbacks(self):
        return self._callbacks
