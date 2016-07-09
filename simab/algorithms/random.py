#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import random

class Random(Algorithm):
    def __init__(self, arms):
        Algorithm.__init__(self, arms)

    def _select_arm(self):
        return random.randrange(len(self.arms))

