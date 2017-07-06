import numpy

# __all__ = ['NumericError', 'ClassificationError', 'BinaryClassificationStats']
__all__ = ['NumericError']


class NumericError:
	def __init__(self, var_name, inf_name=None):
		self.var_name = var_name
		if inf_name == None:
			self.inf_name = "val-" + self.var_name
		else:
			self.inf_name = inf_name
