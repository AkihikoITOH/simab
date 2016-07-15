#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import math
from .algorithm import *

class Softmax(Algorithm):
    """ Softmax(Boltzmann Exploration) Algorithm

    :param tau: decides randomness of the exploration. larger tau means more randomness.
    """
    def __init__(self, arms, tau, label=None, mixture_expected=False):
        Algorithm.__init__(self, arms, label=label, mixture_expected=mixture_expected)
        self.tau = tau
        if self.label is None:
            self.label = 'Softmax(t=%s)' % self.tau

    def _select_arm(self):
        unknown_arm = get_unknown_arm(self.history)
        if unknown_arm is None:
            evals = self._get_evals()
            divider = sum([math.exp(eval_/self.tau) for eval_ in evals ])
            probabilities = [math.exp(eval_/self.tau)/divider for eval_ in evals ]
            selected_arm = pick_by_probability(probabilities)
        else:
            selected_arm = unknown_arm
        return selected_arm

    def summary(self):
        summary = Algorithm.summary(self)
        summary['tau'] = self.tau
        return summary

