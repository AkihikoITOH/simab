#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import math
from .algorithm import *

class Softmax(Algorithm):
    """ Softmax(Boltzmann Exploration) Algorithm

    :param tau: decides randomness of the exploration. larger tau means more randomness.
    """
    def __init__(self, arms, tau, label=None):
        Algorithm.__init__(self, arms, label=label)
        self.tau = tau
        if self.label is None:
            self.label = 'Softmax(t=%s)' % self.tau

    def _select_arm(self):
        unknown_arm = get_unknown_arm(self.history)
        if unknown_arm is None:
            means = get_means(self.history)
            divider = sum([math.exp(mean/self.tau) for mean in means])
            probabilities = [math.exp(mean/self.tau)/divider for mean in means]
            selected_arm = pick_by_probability(probabilities)
        else:
            selected_arm = unknown_arm
        return selected_arm

    def summary(self):
        summary = Algorithm.summary(self)
        summary['tau'] = self.tau

