import numpy

import pdb

# __all__ = ['NumericError', 'ClassificationError', 'BinaryClassificationStats']
__all__ = ['NumericError']

class Inferencer:
    def __init__(self, var_name, inf_name=None):
        if not isinstance(var_name, list):
            self.var_name = [var_name]
        else:
            self.var_name = var_name
        if inf_name == None:
            self.inf_name = "val-" + self.var_name.join(',')
        else:
            self.inf_name = inf_name

    def datapoint(self, output):
        self._datapoint(output)

    def _datapoint(self, output):
        raise NotImplementedError

class NumericError(Inferencer):
    def _datapoint(self, output):
        return output

class ClassificationError(Inferencer):
    def _datapoint(self, output):
        pdb.set_trace()
