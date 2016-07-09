#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from .algorithm import *

class Single(Algorithm):
    """ Keep choosing single arm.

    :param idx_arm: index of arm to keep choosing
    """
    def __init__(self, arms, idx_arm):
        Algorithm.__init__(self, arms)
        self.idx_arm = idx_arm

    def _select_arm(self):
        return self.idx_arm

