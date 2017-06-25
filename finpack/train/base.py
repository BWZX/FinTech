import time
import weakref

from .config import TrainConfig
from ..utils import logger, describe_model
from ..callbacks import Callback, Callbacks

__all__ = ['Trainer']


class Trainer(object):
    """ Base class for a trainer.

    Attributes:
        config (TrainConfig): the config used in this trainer.
        model (ModelDesc)
        sess (tf.Session): the current session in use.
        monitors (Monitors): the monitors. Callbacks can use it for logging.

        epoch_num (int): the number of epochs that have finished.
        local_step (int): the number of steps that have finished in the current epoch.
        global_step (int): the number of steps that have finished.
    """
    # step attr only available after before_train?

    def __init__(self, config):
        """
        Args:
            config (TrainConfig): the train config.
        """
        assert isinstance(config, TrainConfig), type(config)
        self.config = config
        self.model = config.model
        self._input_desc = self.model.get_inputs()

        self.epoch_num = self.config.starting_epoch - 1
        self.local_step = -1

        self._callbacks = []

    def register_callback(self, cb):
        assert isinstance(cb, Callback), cb
        self._callbacks.append(cb)


    def train(self):
        """ Start training """
        self.setup()
        self.main_loop()

    def run_step(self):
        """ Abstract method: run one iteration. Subclass should define what is "iteration".
        """
        raise NotImplementedError

    def _trigger_epoch(self):
        pass

    def setup(self):
        """
        Setup the trainer and be ready for the main loop.
        """
        self._setup()   # subclass will setup the graph

        for cb in self.config.callbacks:
            self.register_callback(cb)

        describe_model(self.model)

        # some final operations that might modify the graph
        logger.info("Setup callbacks ...")
        self._callbacks = Callbacks(self._callbacks)
        self._callbacks.setup_graph(weakref.proxy(self))


    def _setup(self):
        """ setup Trainer-specific stuff for training"""
        raise NotImplementedError

    @property
    def global_step(self):
        return self.config.steps_per_epoch * (self.epoch_num - 1) + \
            self.local_step + 1  # +1: the ongoing step

    def main_loop(self):
        """
        Run the main training loop.
        """
        try:
            self._callbacks.before_train()
            # refresh global step (might have changed by callbacks) TODO ugly
            self._starting_step = self.global_step
            for self.epoch_num in range(
                    self.config.starting_epoch, self.config.max_epoch + 1):
                self._callbacks.before_epoch()
                logger.info("Start Epoch {} ...".format(self.epoch_num))
                start_time = time.time()
                for self.local_step in range(self.config.steps_per_epoch):
                    self.run_step()
                    self._callbacks.trigger_step()
                logger.info("Epoch {} (global_step {}) finished, time:{:.2f} sec.".format(
                    self.epoch_num, self.global_step, time.time() - start_time))

                # trigger epoch outside the timing region.
                self._callbacks.trigger_epoch()
            logger.info("Training has finished!")
        except KeyboardInterrupt:
            logger.info("Detected Ctrl-C and exiting main loop.")
        except:
            raise
        finally:
            self._callbacks.after_train()

