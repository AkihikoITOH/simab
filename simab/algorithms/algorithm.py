#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# TODO: GMMのアームを扱えるようにする

import math
import random
import numpy as np
from sklearn import mixture

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

def get_log_likelihood(series, mean, sd):
    if sd > 0.0:
        first = -float(len(series))/2.0 * math.log(2.0*math.pi*(sd**2.0))
    else:
        first = 0.0
    second = -1.0/(2.0*(sd**2.0))*sum([(mean-s)**2.0 for s in series])
    return first + second

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
    :param mixture_expected: boolean flag. True if expected to have gmm arms.
    """
    def __init__(self, arms, label='Algorithm', mixture_expected=False):
        self.arms = arms
        for arm in self.arms:
            arm.reset()
        # Evaluation of each arm
        self.evals = [None for _ in self.arms]
        # float values for played rounds and None for other rounds.
        self.history = [[] for _ in self.arms]
        # float values for each round.
        self.full_history = [[] for _ in self.arms]
        self.label = label
        self.mixture_expected = mixture_expected

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

    def _get_evals(self):
        means = get_means(self.history)
        sds = get_sds(self.history)

        if not self.mixture_expected:
            return means

        # Calculate evaluations for mixture models
        self.evals_mixture = [{'1': {'means': [mean], 'sds': [sd], 'likelihoods': [get_log_likelihood(hist, mean, sd)], 'populations': [len(hist)], 'likelihood': get_log_likelihood(hist, mean, sd)}} for mean, sd, hist in zip(means, sds, self.history)]
        for idx_arm, (arm, hist) in enumerate(zip(self.arms, self.history)):
            for num_components in [2, 3]:
                gmm = mixture.GMM(n_components=num_components)
                obs = np.array([[sample] for sample in hist])
                gmm.fit(obs)
                weights = gmm.weights_
                means = gmm.means_
                covars = gmm.covars_
                predictions = gmm.predict(obs)
                history_classified = []
                self.evals_mixture[idx_arm][str(num_components)] = {'means': [], 'sds': [], 'likelihoods': [], 'populations': [], 'likelihood': None}
                for i in range(num_components):
                    series = [s for s, p in zip(hist, predictions) if p==i]
                    history_classified.append(series)
                    if len(history_classified) > 1:
                        unbiased_variance = sum([(means[i]-r)**2.0 for r in history_classified])/float(len(history_classified)-1)
                    else:
                        unbiased_variance = sum([(means[i]-r)**2.0 for r in history_classified])/float(len(history_classified))
                    sd = math.sqrt(unbiased_variance)
                    likelihood = get_log_likelihood(series, means[i], sd)
                    population = len(history_classified)
                    self.evals_mixture[idx_arm][str(num_components)]['means'].append(means[i])
                    self.evals_mixture[idx_arm][str(num_components)]['sds'].append(sd)
                    self.evals_mixture[idx_arm][str(num_components)]['likelihoods'].append(likelihood)
                    self.evals_mixture[idx_arm][str(num_components)]['populations'].append(population)
                self.evals_mixture[idx_arm][str(num_components)]['likelihood'] = sum(self.evals_mixture[idx_arm][str(num_components)]['likelihoods'])

        # Decide which model to use (maximum likelihood)
        evals = []
        components = []
        for idx, eval_ in enumerate(self.evals_mixture):
            max_likelihood = max([eval_[str(num_components)]['likelihood'] for num_components in [1, 2, 3]])
            for num_components in [1, 2, 3]:
                if eval_[str(num_components)]['likelihood'] == max_likelihood:
                    components.append(num_components)
                    max_populated_class = idx_max(eval_[str(num_components)]['populations'])
                    mean_of_max_populated_class = eval_[str(num_components)]['means'][max_populated_class]
                    evals.append(mean_of_max_populated_class)

        # Generate list of evaluations of arms
        return evals

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
        summary['total_reward'] = sum([sum(h) for h in get_dense_history(self.history)])
        summary['plays'] = [len(h) for h in get_dense_history(self.history)]
        summary['true_means'] = [arm.mu if not hasattr(arm, 'gmm') else arm.mus for arm in self.arms]
        summary['true_sds'] = [arm.sigma if not hasattr(arm, 'gmm') else arm.sigmas  for arm in self.arms]
        summary['empirical_means'] = get_means(self.history)
        summary['empirical_sds'] = get_sds(self.history)
        if self.mixture_expected:
            summary['evals_mixture'] = self.evals_mixture
        is_every_arm_predicted = all([arm.is_predicted for arm in self.arms])
        if is_every_arm_predicted:
            summary['predictions'] = [arm.prediction for arm in self.arms]
        return summary


