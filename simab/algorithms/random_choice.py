#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import random
from .algorithm import *

class Random(Algorithm):
    def __init__(self, arms, label='Random'):
        Algorithm.__init__(self, arms, label=label)

    def _select_arm(self):
        return random.randrange(len(self.arms))

