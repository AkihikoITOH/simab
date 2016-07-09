#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import random

def idx_max(list_):
    return list_.index(max(list_))

def pick_by_probability(list_):
    r = random.random()
    pick = None
    for idx, p in enumerate(list_):
        r -= p
        if r <= 0.0:
            pick = idx
            break
    return pick

def get_dense_history(history):
    return [[v for v in hist_arm if v is not None] for hist_arm in history]

class Algorithm(object):
    """ Abstract class for various Multi-Armed Bandit algorithms.

    :param arms: list of arms
    """
    def __init__(self, arms):
        self.arms = arms
        # Number of plays of each arm
        self.counts = [0 for _ in self.arms]
        # Evaluation of each arm
        self.evals = [None for _ in self.arms]
        # float values for played rounds and None for other rounds.
        self.history = [[] for _ in self.arms]
        # float values for each round.
        self.full_history = [[] for _ in self.arms]

    def _select_arm(self):
        return idx_arm

    def _update(self, selected_arm, reward):
        self.counts[selected_arm] += 1

        for idx, value in enumerate(self.values):
            self.full_history[idx].append(value)
            if selected_arm == idx:
                self.history[idx].append(value)
            else:
                self.history[idx].append(None)

    def play(self, dry=False):
        selected_arm = self._select_arm()
        arm = self.arms[selected_arm]
        reward = arm.pick()

        if not dry:
            self._update(selected_arm, reward)

        return selected_arm, reward

