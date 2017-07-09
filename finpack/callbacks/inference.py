import numpy as np

import pdb

# __all__ = ['NumericError', 'ClassificationError', 'BinaryClassificationStats']
__all__ = ['NumericError', 'ClassificationError']

class Inferencer:
    def __init__(self, var_names, inf_name=None):
        if not isinstance(var_names, list):
            self.var_names = [var_names]
        else:
            self.var_names = var_names
        if inf_name == None:
            self.inf_name = "val-" + ",".join(self.var_names)
        else:
            self.inf_name = inf_name

    def datapoint(self, output):
        return self._datapoint(output)

    def _datapoint(self, output):
        raise NotImplementedError

class NumericError(Inferencer):
    def _datapoint(self, output):
        return output.cpu().numpy()

class ClassificationError(Inferencer):
    def _datapoint(self, output):
        pred = np.argmax(output[0].cpu().numpy(), 1)
        label = output[1].cpu().numpy()
        return (pred == label).astype(float)
