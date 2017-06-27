
__all__ = ['Callback', 'ProxyCallback']


class Callback(object):
    """ Base class for all callbacks.

    Attributes:
        epoch_num(int): the number of the current epoch.
        global_step(int): the number of global steps that have finished.
        local_step(int): the local steps within the current epoch.
        trainer(Trainer): the trainer.
        graph(tf.Graph): the graph.

    Note:
        These attributes are available only after (and including)
        :meth:`_setup_graph`.

    .. document private functions
    .. automethod:: _before_train
    .. automethod:: _before_run
    .. automethod:: _after_run
    .. automethod:: _trigger_step
    .. automethod:: _trigger_epoch
    .. automethod:: _trigger
    .. automethod:: _after_train
    """

    def setup_graph(self, trainer):
        self._steps_per_epoch = trainer.config.steps_per_epoch
        self.trainer = trainer
        self._setup_graph()

    def _setup_graph(self):
        pass

    def before_train(self):
        self._before_train()

    def _before_train(self):
        pass
        # self._steps_per_epoch = trainer.config.steps_per_epoch
        # self.trainer = trainer

    def before_epoch(self):
        self._before_epoch()

    def _before_epoch(self):
        pass

    def trigger_step(self):
        self._trigger_step()

    def _trigger_step(self):
        """
        Called after each :meth:`Trainer.run_step()` completes. Defaults to no-op.

        You can override it to implement, e.g. a ProgressBar.
        """
        pass

    def trigger_epoch(self):
        self._trigger_epoch()

    def _trigger_epoch(self):
        """
        Called after the completion of every epoch. Defaults to call ``self.trigger()``
        """
        self.trigger()

    def trigger(self):
        self._trigger()

    def _trigger(self):
        """
        Override this method to define a general trigger behavior, to be used with trigger schedulers.
        Note that the schedulers (e.g. :class:`PeriodicTrigger`) might call this
        method both inside an epoch and after an epoch.

        When used without the scheduler, this method by default will be called by `trigger_epoch()`.
        """
        pass

    def after_train(self):
        self._after_train()

    def _after_train(self):
        """
        Called after training.
        """
        pass

    @property
    def epoch_num(self):
        return self.trainer.epoch_num

    @property
    def global_step(self):
        return self.trainer.global_step

    @property
    def local_step(self):
        return self.trainer.local_step

    def __str__(self):
        return type(self).__name__


class ProxyCallback(Callback):
    """ A callback which proxy all methods to another callback.
        It's useful as a base class of callbacks which decorate other callbacks.
    """

    def __init__(self, cb):
        """
        Args:
            cb(Callback): the underlying callback
        """
        assert isinstance(cb, Callback), type(cb)
        self.cb = cb

    def _before_train(self):
        self.cb.before_train()

    def _setup_graph(self):
        self.cb.setup_graph(self.trainer)

    def _trigger_epoch(self):
        self.cb.trigger_epoch()

    def _trigger_step(self):
        self.cb.trigger_step()

    def _after_train(self):
        self.cb.after_train()

    def _before_epoch(self):
        self.cb.before_epoch()

    def _after_epoch(self):
        self.cb.after_epoch()

    def __str__(self):
        return "Proxy-" + str(self.cb)