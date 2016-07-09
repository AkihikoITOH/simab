#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from .algorithm import *

class Single(Algorithm):
    """ Keep choosing single arm.

    :param idx_arm: index of arm to keep choosing
    """
    def __init__(self, arms, idx_arm, label=None):
        Algorithm.__init__(self, arms, label=None)
        self.idx_arm = idx_arm
        if self.label is None:
            self.label = 'Single(%s)' % self.arms[self.idx_arm].label

    def _select_arm(self):
        return self.idx_arm

