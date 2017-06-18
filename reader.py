import tushare as ts

from abc import abstractmethod, ABCMeta
import six
import pdb

from torch.autograd import Variable
import torch

@six.add_metaclass(ABCMeta)
class DataFlow(object):
    """ Base class for all DataFlow"""

    @abstractmethod
    def get_data(self):
        """
            The method to generate datapoints.

            Yields:
                list: The datapoint, i.e. list of components.
        """

    def size(self):
        """
        Returns:
            int: size of this data flow.

        Raises:
            :class:`NotImplementedError` if this DataFlow doesn't have a size.
        """
        raise NotImplementedError()

    def reset_state(self):
        """
        Reset state of the dataflow. It has to be called before producing datapoints.

        For example, RNG **has to** be reset if used in the DataFlow,
        otherwise it won't work well with prefetching, because different
        processes will have the same RNG state.
        """
        pass



class RNGDataFlow(DataFlow):
    """ A DataFlow with RNG"""

    def reset_state(self):
        """ Reset the RNG """
        self.rng = get_rng(self)



class StockHistory(DataFlow):
    def __init__(self, stock_list, start, end, columns=["close"], seq_len=25, batch_size=1):
        if isinstance(stock_list, list) == False:
            stock_list = [stock_list]
        self.stock_list = stock_list
        self.start = start
        self.end = end
        if isinstance(columns, list) == False:
            columns=columns.split(',')
        self.columns=columns
        self.seq_len=seq_len
        self.batch_size=batch_size

        # get data and save with pickle
        self.seq_list = []
        for stock in stock_list:
            stock_data = ts.get_k_data(stock, start=start, end=end)
            target_data = stock_data[columns].as_matrix()
            self.seq_list.append(target_data)


    def get_data(self):

        for seq in self.seq_list:
            reset = True
            idx = 0
            while True:
                if (idx + self.seq_len + 1 >= seq.shape[0]):
                    break
                input = Variable(torch.FloatTensor(seq[idx: idx + self.seq_len]))
                label = Variable(torch.FloatTensor(seq[idx + 1: idx + self.seq_len + 1]))
                input = input.unsqueeze(1)
                label = label.unsqueeze(1)
                yield ([input, label], reset)
                idx += self.seq_len
                reset = False

