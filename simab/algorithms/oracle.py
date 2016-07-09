#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from .algorithm import *

class Oracle(Algorithm):
    """ Oracle - Choose best arm for each round.
    """
    def __init__(self, arms, label='Oracle'):
        Algorithm.__init__(self, arms, label=label)

    def _select_arm(self):
        return idx_max([arm.prediction[arm.count] for arm in self.arms])


