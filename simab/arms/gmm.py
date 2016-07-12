#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import random
import numpy as np
from sklearn import mixture
from .arm import Arm

# TODO: 実装する

class GMMArm(Arm):
    """ Arm with GMM distribution model

    :param mus: list of means for each mixture component
    :param sigmas: list of standard deviations for each mixture component
    :param weights: list of weights for each mixture component
    """
    def __init__(self, mus, sigmas, weights, truncate=[0.0, 1.0], label=None):
        #平均mu, 標準偏差stdのガウス分布に従う標本をN個生成する
        Arm.__init__(self, label)
        self.num_mix = len(mus)
        self.mus = mus
        self.sigmas = sigmas
        self.weights = weights
        self.truncate = truncate
        if sum(self.weights) != 1.0:
            'Sum of weights should be 1.0.'
            return
        self.gmm = mixture.GMM(n_components=self.num_mix)
        samples = []
        for i in range(self.num_mix):
            samples += [random.gauss(self.mus[i], self.sigmas[i]) for _ in range(weights[i]*10000)]
        obs = np.array([[sample] for sample in samples if sample>=self.truncate[0] and sample<=self.truncate[1]])
        self.gmm.fit(obs)

    def _is_valid_reward(self, reward):
        return self.truncate is None or (reward>=self.truncate[0] and reward<=self.truncate[1])

    def pick(self, dry=False):
        reward = Arm.pick(self, dry)
        # If prediction exists, simply use it.
        while reward is None or not self._is_valid_reward(reward):
            reward = self.gmm.sample(1)
        return reward


