import numpy as np

from .base import DataFlow, ProxyDataFlow, RNGDataFlow

__all__ = ["BatchData"]

class BatchData(ProxyDataFlow):

    def __init__(self, ds, batch_size, remainder=False, stack_dim=0):
        """
        Args:
            ds (DataFlow): Its components must be either scalars or :class:`np.ndarray`.
                Each component has to be of the same shape across datapoints.
            batch_size(int): batch size
            remainder (bool): whether to return the remaining data smaller than a batch_size.
                If set True, it will possibly generates a data point of a smaller batch size.
                Otherwise, all generated data are guranteed to have the same size.
        """
        super(BatchData, self).__init__(ds)
        if not remainder:
            try:
                assert batch_size <= ds.size()
            except NotImplementedError:
                pass
        self.batch_size = batch_size
        self.remainder = remainder
        self.stack_dim = stack_dim


    def size(self):
        ds_size = self.ds.size()
        div = ds_size // self.batch_size
        rem = ds_size % self.batch_size

        if rem == 0:
            return div
        return div + int(self.remainder)

    def get_data(self):
        """
        Yields:
            Batched data by stacking each component on the stack_dim.
        """
        holder = []
        for data in self.ds.get_data():
            holder.append(data)
            if len(holder) == self.batch_size:
                yield BatchData._aggregate_batch(holder, self.stack_dim)
                del holder[:]
        if self.remainder and len(holder) > 0:
            yield BatchData._aggregate_batch(holder, self.stack_dim)

    @staticmethod
    def _aggregate_batch(data_holder, stack_dim):
        size = len(data_holder[0])
        result = []
        for k in range(size):
            dt = data_holder[0][k]
            if type(dt) in [int, bool]:
                tp = 'int32'
            elif type(dt) == float:
                tp = 'float32'
            else:
                try:
                    tp = dt.dtype
                except:
                    raise TypeError("Unsupported type to batch: {}".format(type(dt)))
            data = [x[k] for x in data_holder]
            result.append(np.concatenate(data, axis=stack_dim))
            # result.append(np.asarray([x[k] for x in data_holder], dtype=tp))
        return result
