#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from simab.arms.normal import NormalArm
from simab.algorithms.random_choice import Random
from simab.algorithms.oracle import Oracle
from simab.algorithms.epsilon_first import EpsilonFirst
from simab.algorithms.epsilon_greedy import EpsilonGreedy
from simab.algorithms.softmax import Softmax
from simab.algorithms.ucb1 import UCB1
from simab.algorithms.single_arm import Single

class MultipleSimsOfAlgorithm(object):
    def __init__(self, sample_algorithm, rounds, num_sims):
        class_algo = type(sample_algorithm)
        self.algos = []
        self.num_sims = num_sims
        self.rounds = rounds
        for _ in range(self.num_sims):
            arms = [NormalArm(arm.mu, arm.sigma, label=arm.label) for arm in sample_algorithm.arms]
            for arm in arms:
                arm.predict(self.rounds)
            kwargs = {}
            kwargs['label'] = sample_algorithm.label
            if class_algo == EpsilonFirst:
                kwargs['epsilon'] = sample_algorithm.epsilon
            elif class_algo == EpsilonGreedy:
                kwargs['epsilon'] = sample_algorithm.epsilon
            elif class_algo == Softmax:
                kwargs['tau'] = sample_algorithm.tau
            elif class_algo == Single:
                kwargs['idx_arm'] = sample_algorithm.idx_arm
            else:
                pass
            self.algos.append(class_algo(arms, **kwargs))

    def simulation(self):
        for algo in self.algos:
            for _ in range(self.rounds):
                algo.play()

    def summary(self, full=False):
        summary = {}
        one_algo = self.algos[0].summary()
        if full:
            summary['simulations'] = [algo.summary() for algo in self.algos]
        summary['algorithm'] = one_algo['algorithm']
        summary['history'] = one_algo['history']
        summary['true_means'] = one_algo['true_means']
        summary['true_sds'] = one_algo['true_sds']
        summary['total_reward'] = float(sum([algo.summary()['total_reward'] for algo in self.algos]))/float(len(self.algos))
        summary['plays'] = one_algo['plays']
        if 'predictions' in one_algo:
            summary['predictions'] = one_algo['predictions']
        return summary

