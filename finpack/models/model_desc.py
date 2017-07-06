from ..utils import logger
from torch.autograd import Variable

__all__ = ['ModelDesc']

class ModelDesc(object):

    def __init__(self):
        self.summaries = { }
        self.inf_summaries = { }
        self.hyper_params = { }
        self.is_train = None

    def run_graph(self, inputs):
        self._run_graph(inputs)
        self.add_summary(self.hyper_params["learning_rate"], "learning_rate")

    def _run_graph(self, inputs):
        pass

    def get_cost(self):
        cost = self._get_cost()
        return cost

    def _get_cost(self, *args):
        return self.cost

    def train(self):
        self._train()
        self.is_train = True

    def _train(self):
        pass

    def eval(self):
        self._eval()
        self.is_train = False

    def _eval(self):
        pass

    def add_summary(self, variable, name):
        if self.is_train == False:
            return
        if name not in self.summaries.keys():
            self.summaries[name] = []
        if isinstance(variable, Variable):
            variable = variable.data[0]
        self.summaries[name].append(variable)

    def clear_summaries(self):
        self.summaries = { }

    def get_optimizer(self):
        self.optimizer = self._get_optimizer()
        self.hyper_params["learning_rate"] = self.optimizer.param_groups[0]['lr']
        return self.optimizer

    def _get_optimizer(self):
        raise NotImplementedError()
