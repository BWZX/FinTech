import tqdm
from ..dataflow import DataFlow

from .base import Callback
# from .inference import Inferencer

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

