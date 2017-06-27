from ..utils import logger

__all__ = ['ModelDesc']


class ModelDesc(object):

    def __init__(self):
        self.summaries = { }
        self.hyper_params = { }
        self.hyper_params["learning_rate"] = 0

    def get_cost(self):
        cost = self._get_cost()
        return cost

    def _get_cost(self, *args):
        return self.cost

    def add_summary(self, variable, name):
        if name not in self.summaries.keys():
            self.summaries[name] = []
        self.summaries[name].append(variable.data[0])

    def clear_summaries(self):
        self.summaries = { }

    def get_optimizer(self):
        self.optimizer = self._get_optimizer()
        return self.optimizer

    def _get_optimizer(self):
        raise NotImplementedError()
