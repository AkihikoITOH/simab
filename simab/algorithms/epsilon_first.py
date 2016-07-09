#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import random
from .algorithm import *

class EpsilonFirst(Algorithm):
    """ Epsilon-First Algorithm

    :param epsilon:
    :param rounds: total rounds
    """
    def __init__(self, arms, epsilon, rounds=None):
        Algorithm.__init__(self, arms)
        self.epsilon = epsilon
        if rounds is None:
            self.rounds = len(arms[0].prediction)
        else:
            self.rounds = rounds
        self.exploration_rounds = int(self.epsilon * float(self.rounds))

    def _rounds_so_far(self):
        return len(self.history[0])

    def _select_arm(self):
        if self._rounds_so_far() <= self.exploration_rounds:
            selected_arm = random.randrange(len(self.arms))
        else:
            selected_arm = idx_max(get_means(self.history))
        return selected_arm

