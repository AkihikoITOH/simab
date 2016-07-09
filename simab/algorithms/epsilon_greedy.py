#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from .algorithm import *

class EpsilonGreedy(Algorithm):
    """ Epsilon-Greedy Algorithm

    :param epsolon:
    """
    def __init__(self, arms, epsilon):
        Algorithm.__init__(self, arms)
        self.epsilon = epsilon

    def _select_arm(self):
        probabilities = [None for _ in self.arms]
        best_arm = idx_max(get_means(self.history))
        for i in range(len(self.arms)):
            if i == best_arm:
                probabilities[i] = 1.0 - self.epsilon + self.epsilon/float(len(self.arms))
            else:
                probabilities[i] = self.epsilon/float(len(self.arms))
        return pick_by_probability(probabilities)

