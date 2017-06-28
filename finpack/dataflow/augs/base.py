from ...utils import get_rng

__all__ = ['Augmentor', 'AugmentorList']


class Augmentor(object):
    """ Base class for an augmentor"""

    def __init__(self):
        self.reset_state()

    def reset_state(self):
        self.rng = get_rng(self)

    def augment(self, d):
        d = self._augment(d)
        return d

    def _augment(self, d):
        pass

    def _rand_range(self, low=1.0, high=None, size=None):
        if high is None:
            low, high = 0, low
        if size is None:
            size = []
        return self.rng.uniform(low, high, size)


class AugmentorList(Augmentor):
    """
    Augment by a list of augmentors
    """

    def __init__(self, augmentors):
        """
        Args:
            augmentors (list): list of :class:`ImageAugmentor` instance to be applied.
        """
        self.augs = augmentors
        super(AugmentorList, self).__init__()


    def _augment(self, dp):
        for aug in self.augs:
            dp = aug._augment(dp)
        return dp

    def reset_state(self):
        """ Will reset state of each augmentor """
        for a in self.augs:
            a.reset_state()
