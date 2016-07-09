#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from .algorithm import *

class EpsilonGreedy(Algorithm):
    """ Epsilon-Greedy Algorithm

    :param epsilon:
    """
    def __init__(self, arms, epsilon, label=None):
        Algorithm.__init__(self, arms, label=label)
        self.epsilon = epsilon
        if self.label is None:
            self.label = 'Epsilon Greedy(e=%s)' % self.epsilon

    def _select_arm(self):
        probabilities = [None for _ in self.arms]
        best_arm = idx_max(get_means(self.history))
        for i in range(len(self.arms)):
            if i == best_arm:
                probabilities[i] = 1.0 - self.epsilon + self.epsilon/float(len(self.arms))
            else:
                probabilities[i] = self.epsilon/float(len(self.arms))
        return pick_by_probability(probabilities)

    def summary(self):
        summary = Algorithm.summary(self)
        summary['epsilon'] = self.epsilon

