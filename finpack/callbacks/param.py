#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: param.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import os
import operator
import tensorflow as tf

from .base import Callback
from ..utils import logger

__all__ = ['LearningRateSetter', 'HyperParamSetter', 'HumanHyperParamSetter',
           'ScheduledHyperParamSetter', 'HyperParamSetterWithFunc',
           ]

class LearningRateSetter(Callback):

    def _setup_graph(self):
        self.optimizer = self.trainer.model.optimizer

    def _trigger(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.trainer.model.hyper_params["learning_rate"]


class HyperParamSetter(Callback):
    """
    An abstract base callback to set hyperparameters.
    """

    def __init__(self, param):
        self.param = param
        self.last_value = None

    # def _setup_graph(self):
    #     self.param.setup_graph()

    def get_value_to_set(self):
        """
        Returns:
            The value to assign to the variable.

        Note:
            Subclasses will implement the abstract method
            :meth:`_get_value_to_set`, which should return a new value to
            set, or return None to do nothing.
        """
        ret = self._get_value_to_set()
        if ret is not None and ret != self.last_value:
            logger.info("{} at epoch {} will change to {:.8f}".format(
                self.param, self.epoch_num + 1, ret))
        self.last_value = ret
        return ret

    def _get_value_to_set(self):
        pass

    def get_current_value(self):
        """
        Returns:
            The current value of the param.
        """
        return self.trainer.model.hyper_params[self.param]

    def _trigger(self):
        self._set_param()

    def _before_train(self):
        self._set_param()

    def _set_param(self):
        self.trainer.model.hyper_params[self.param] = self.get_value_to_set()


class HumanHyperParamSetter(HyperParamSetter):
    """
    Set hyperparameter by loading the value from a file each time it get called.
    This is useful for manually tuning some parameters (e.g. learning_rate)
    without interrupting the training.
    """

    def __init__(self, param, file_name='hyper.txt'):
        """
        Args:
            param: same as in :class:`HyperParamSetter`.
            file_name(str): a file containing the new value of the parameter.
                Each line in the file is a ``k:v`` pair, for example, ``learning_rate:1e-4``.
                If the pair is not found, the param will not be changed.
        """
        super(HumanHyperParamSetter, self).__init__(param)
        self.file_name = os.path.join(logger.LOG_DIR, file_name)
        logger.info("Use {} to control hyperparam {}.".format(
            self.file_name, self.param.readable_name))

    def _get_value_to_set(self):
        # ignore if no such file exists
        if not os.path.isfile(self.file_name):
            return self.get_current_value()
        try:
            with open(self.file_name) as f:
                lines = f.readlines()
            lines = [s.strip().split(':') for s in lines]
            dic = {str(k): float(v) for k, v in lines}
            ret = dic[self.param.readable_name]
            return ret
        except:
            logger.warn(
                "Cannot find {} in {}".format(
                    self.param.readable_name, self.file_name))
            return self.get_current_value()


class ScheduledHyperParamSetter(HyperParamSetter):
    """
    Set hyperparameters by a predefined epoch-based schedule.
    """

    def __init__(self, param, schedule, interp=None):
        """
        Args:
            param: same as in :class:`HyperParamSetter`.
            schedule (list): with the format ``[(epoch1, val1), (epoch2, val2), (epoch3, val3)]``.
                Each ``(ep, val)`` pair means to set the param
                to "val" __after__ the completion of epoch `ep`.
                If ep == 0, the value will be set before the first epoch
                (by default the first is epoch 1).
            interp: None: no interpolation. 'linear': linear interpolation

        Example:
            .. code-block:: python

                ScheduledHyperParamSetter('learning_rate',
                                          [(30, 1e-2), (60, 1e-3), (85, 1e-4), (95, 1e-5)]),
        """
        schedule = [(int(a), float(b)) for a, b in schedule]
        self.schedule = sorted(schedule, key=operator.itemgetter(0))
        if interp is not None:
            assert interp == 'linear'
        self.interp = interp
        super(ScheduledHyperParamSetter, self).__init__(param)

    def _get_value_to_set(self):
        if self.interp is None:
            for e, v in self.schedule:
                if e == self.epoch_num:
                    return v
            return self.get_current_value()
        else:
            laste, lastv = None, None
            for e, v in self.schedule:
                if e == self.epoch_num:
                    return v
                if e > self.epoch_num:
                    break
                laste, lastv = e, v
            if laste is None or laste == e:
                # hasn't reached the first scheduled point, or reached the end of all scheduled points
                return self.get_current_value()
            v = (self.epoch_num - laste) * 1. / (e - laste) * (v - lastv) + lastv
            return v


class HyperParamSetterWithFunc(HyperParamSetter):
    """ Set the parameter by a function of epoch num and old value. """
    def __init__(self, param, func):
        """
        Args:
            param: same as in :class:`HyperParamSetter`.
            func: ``param`` will be set by ``new_value = func(epoch_num, old_value)``.
                ``epoch_num`` is the number of epochs that have finished.

        Example:
            Decrease by a factor of 0.9 every two epochs:

            .. code-block:: python

                HyperParamSetterWithFunc('learning_rate',
                                         lambda e, x: x * 0.9 if e % 2 == 0 else x)
        """
        super(HyperParamSetterWithFunc, self).__init__(param)
        self.f = func

    def _get_value_to_set(self):
        return self.f(self.epoch_num, self.get_current_value())


