#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import math
from .algorithm import *

def _bonus(rounds, plays):
    return math.sqrt(2.0*math.log(float(rounds))/float(plays))

class UCB1(Algorithm):
    """ Upper Confidence Bounds Algorithm
    """
    def __init__(self, arms, label='UCB1', mixture_expected=False):
        Algorithm.__init__(self, arms, label=label, mixture_expected=mixture_expected)

    def _rounds_so_far(self):
        return len(self.history[0])

    def _select_arm(self):
        unknown_arm = get_unknown_arm(self.history)
        if unknown_arm is None:
            evals = self._get_evals()
            plays = [len(hist) for hist in get_dense_history(self.history)]
            rounds = self._rounds_so_far()

            evaluations = [evals[idx] + _bonus(rounds, plays[idx]) for idx in range(len(self.arms))]
            selected_arm = idx_max(evaluations)
        else:
            selected_arm = unknown_arm
        return selected_arm

