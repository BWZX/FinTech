import numpy

import pdb

# __all__ = ['NumericError', 'ClassificationError', 'BinaryClassificationStats']
__all__ = ['NumericError']

class Inferencer:
    def __init__(self, var_names, inf_name=None):
        if not isinstance(var_names, list):
            self.var_names = [var_names]
        else:
            self.var_names = var_namse
        if inf_name == None:
            self.inf_name = "val-" + self.var_names.join(',')
        else:
            self.inf_name = inf_name

    def datapoint(self, output):
        self._datapoint(output)

    def _datapoint(self, output):
        raise NotImplementedError

class NumericError(Inferencer):
    def _datapoint(self, output):
        return output[0]

class ClassificationError(Inferencer):
    def _datapoint(self, output):
        pdb.set_trace()
