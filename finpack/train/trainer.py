from .base import Trainer

from ..utils import logger

from torch.autograd import Variable

__all__ = ['SimpleTrainer']


class SimpleTrainer(Trainer):
    """ A naive demo trainer which iterates over a DataFlow and run the
    graph. It's not efficient compared to QueueInputTrainer or others."""

    def __init__(self, config):
        super(SimpleTrainer, self).__init__(config)
        self.ds = config.dataflow

    def run_step(self):
        self.train_op.zero_grad()

        dp = next(self.ds.get_data())
        assert len(dp) == len(self._input_desc), "Length of inputs should be same with length of input_desc"
        dp = [Variable(self._input_desc[idx](ele).cuda()) for idx, ele in enumerate(dp)]

        self.model.run_graph(dp)
        self.model.cost.backward()

        self.train_op.step()

    def _setup(self):
        self.ds.reset_state()
        self.train_op = self.model.get_optimizer()
