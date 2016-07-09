#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import random
from .arm import Arm

class NormalArm(Arm):
    """ Arms which return rewards from normal distributions.

    :param mu: the mean of the distribution.
    :param sigma: the variance of the distribution
    :param truncate: interval of truncated distribution. Given `None`, no truncate.
    :param label: label of the arm
    """
    def __init__(self, mu, sigma, truncate=[0.0, 1.0], label=None):
        Arm.__init__(self, label)
        self.mu = mu
        self.sigma = sigma
        self.truncate = truncate

    def _is_valid_reward(self, reward):
        return self.truncate is None or (reward>=self.truncate[0] and reward<=self.truncate[1])

    def pick(self):
        reward = Arm.pick(self)
        # If prediction exists, simply use it.
        while reward is None or not self._is_valid_reward(reward):
            reward = random.gauss(self.mu, self.sigma)
        return reward

