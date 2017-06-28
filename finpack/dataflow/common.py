import numpy as np
import copy as copy_mod
from .augs import AugmentorList

from .base import DataFlow, ProxyDataFlow, RNGDataFlow
from ..utils import logger

__all__ = ["BatchData", "MapData", "AugmentData"]

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

class MapData(ProxyDataFlow):
    """ Apply a mapper/filter on the DataFlow"""

    def __init__(self, ds, func):
        """
        Args:
            ds (DataFlow): input DataFlow
            func (datapoint -> datapoint | None): takes a datapoint and returns a new
                datapoint. Return None to discard this data point.
                Note that if you use the filter feature, ``ds.size()`` will be incorrect.

        Note:
            Please make sure func doesn't modify the components
            unless you're certain it's safe.
        """
        super(MapData, self).__init__(ds)
        self.func = func

    def get_data(self):
        for dp in self.ds.get_data():
            ret = self.func(dp)
            if ret is not None:
                yield ret


class AugmentData(MapData):
    """
    Apply image augmentors on 1 component.
    """
    def __init__(self, ds, augmentors, copy=True):
        """
        Args:
            ds (DataFlow): input DataFlow.
            augmentors (AugmentorList): a list of :class:`imgaug.ImageAugmentor` to be applied in order.
            index (int): the index of the image component to be augmented.
            copy (bool): Some augmentors modify the input images. When copy is
                True, a copy will be made before any augmentors are applied,
                to keep the original images not modified.
                Turn it off to save time when you know it's OK.
        """
        if isinstance(augmentors, AugmentorList):
            self.augs = augmentors
        else:
            self.augs = AugmentorList(augmentors)

        self._nr_error = 0

        def func(x):
            try:
                if copy:
                    x = copy_mod.deepcopy(x)
                ret = self.augs.augment(x)
            except KeyboardInterrupt:
                raise
            except Exception:
                self._nr_error += 1
                if self._nr_error % 1000 == 0 or self._nr_error < 10:
                    logger.exception("Got {} augmentation errors.".format(self._nr_error))
                return None
            return ret

        super(AugmentData, self).__init__(
            ds, func)

    def reset_state(self):
        self.ds.reset_state()
        self.augs.reset_state()
