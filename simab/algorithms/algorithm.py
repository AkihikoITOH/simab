#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import math
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

def get_means(history):
    dense_history = get_dense_history(history)
    return [sum(h)/float(len(h)) if len(h)>0 else None for h in dense_history]

def get_unbiased_variances(history):
    means = get_means(history)
    dh = get_dense_history(history)
    vs = [sum([(means[idx]-r)**2.0 for r in h])/float(len(h)-1) if len(h)>1 else 0.0 for idx, h in enumerate(dh)]
    return vs

def get_sds(history):
    unbiased_variances = get_unbiased_variances(history)
    return [math.sqrt(uv) for uv in unbiased_variances]

def get_unknown_arm(history):
    unknown = None
    for idx, hist_of_arm in enumerate(get_dense_history(history)):
        if len(hist_of_arm) == 0:
            unknown = idx
            break
    return unknown

class Algorithm(object):
    """ Abstract class for various Multi-Armed Bandit algorithms.

    :param arms: list of arms
    :param label: label of the arm
    """
    def __init__(self, arms, label='Algorithm'):
        self.arms = arms
        # Evaluation of each arm
        self.evals = [None for _ in self.arms]
        # float values for played rounds and None for other rounds.
        self.history = [[] for _ in self.arms]
        # float values for each round.
        self.full_history = [[] for _ in self.arms]
        self.label = label

    def _select_arm(self):
        return selected_arm

    def _update(self, selected_arm, reward):
        for idx, arm in enumerate(self.arms):
            if selected_arm == idx:
                self.history[idx].append(reward)
            else:
                self.history[idx].append(None)
            # increment each arm's count if its rewards are predicted.
            if arm.is_predicted and idx!=selected_arm:
                arm.count += 1

    def play(self, dry=False):
        selected_arm = self._select_arm()
        arm = self.arms[selected_arm]
        reward = arm.pick()

        if not dry:
            self._update(selected_arm, reward)

        return selected_arm, reward

    def summary(self):
        summary = {}
        summary['algorithm'] = self.label
        summary['history'] = self.history
        summary['true_means'] = [arm.mu for arm in self.arms]
        summary['true_sds'] = [arm.sigma for arm in self.arms]
        summary['empirical_means'] = get_means(self.history)
        summary['empirical_sds'] = get_sds(self.history)
        summary['total_reward'] = sum([sum(h) for h in get_dense_history(self.history)])
        summary['plays'] = [len(h) for h in get_dense_history(self.history)]
        return summary


