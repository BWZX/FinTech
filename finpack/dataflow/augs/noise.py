from .base import Augmentor
import numpy as np

__all__ = ['GaussianNoise']

class GaussianNoise(Augmentor):
    """
    Add random Gaussian noise N(0, sigma^2) of the same shape to seq.
    """
    def __init__(self, sigma=1):
        super(GaussianNoise, self).__init__()
        self.sigma = sigma

    def _augment(self, dp):
        noise = self.rng.randn(*dp[0].shape)
        old_dtype = dp[0].dtype
        dp[0] = dp[0] + noise * self.sigma
        dp[0] = dp[0].astype(old_dtype)
        return dp
