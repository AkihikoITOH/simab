#!/usr/bin/env python
# -*- encoding: utf-8 -*-

class EpsilonGreedy(Algorithm):
    def __init__(self, arms, epsilon):
        Algorithm.__init__(self, arms)
        self.epsilon = epsilon



