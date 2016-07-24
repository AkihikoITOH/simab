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
    # print 'series: ', series
    # print 'mean: ', mean
    # print 'sd: ', sd
    # TODO: ここどうすべきか
    if sd!=0.0:
        first = -float(len(series))/2.0 * math.log(2.0*math.pi*(sd**2.0))
    else:
        first = 0.0
    if sd!=0.0 and len(series)>0:
        second = -1.0/(2.0*(sd**2.0))*sum([(mean-s)**2.0 for s in series])
    else:
        second = 0.0
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
        self.evals = []
        self.full_evals = []
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
        means_of_arms = get_means(self.history)
        sds_of_arms = get_sds(self.history)
        self.evals_mixture = []
        dense_history = get_dense_history(self.history)
        for _ in self.arms:
            e = {str(i): {'means': [], 'sds': [], 'likelihoods': [], 'populations': [], 'likelihood': None} for i in [1, 2, 3]}
            self.evals_mixture.append(e)

        for idx, (mix, arm) in enumerate(zip(self.evals_mixture, self.arms)):
            mix['1']['means'].append(means_of_arms[idx])
            mix['1']['sds'].append(sds_of_arms[idx])
            mix['1']['likelihoods'].append(get_log_likelihood(dense_history[idx], means_of_arms[idx], sds_of_arms[idx]))
            mix['1']['populations'].append(len(dense_history[idx]))
            mix['1']['likelihood'] = get_log_likelihood(dense_history[idx], means_of_arms[idx], sds_of_arms[idx])

            for n_components in [2, 3]:
                if not self.mixture_expected:
                    break
                is_valid = []
                for hist in dense_history:
                    is_valid.append(len(hist) > 2*n_components)
                if not all(is_valid):
                    continue
                gmm = mixture.GMM(n_components=n_components)
                obs = np.array([[sample] for sample in dense_history[idx]])
                gmm.fit(obs)
                weights = gmm.weights_.tolist()
                means = [m[0] for m in gmm.means_.tolist()]
                covars = [c[0] for c in gmm.covars_.tolist()]
                predictions = gmm.predict(obs).tolist()
                for i in range(n_components):
                    series = [s for s, p in zip(dense_history[idx], predictions) if int(p)==i]
                    mean = means[i]
                    if len(series) > 1:
                        unbiased_variance = sum([(mean-r)**2.0 for r in series])/float(len(series)-1)
                    else:
                        unbiased_variance = 0.0
                    sd = math.sqrt(unbiased_variance)
                    population = len(series)
                    mix[str(n_components)]['means'].append(mean)
                    mix[str(n_components)]['sds'].append(sd)
                    mix[str(n_components)]['likelihoods'].append(get_log_likelihood(series, mean, sd))
                    mix[str(n_components)]['populations'].append(len(series))
                mix[str(n_components)]['likelihood'] = sum(mix[str(n_components)]['likelihoods'])

        # Decide which model to use (maximum likelihood)
        evals = []
        for idx, mix in enumerate(self.evals_mixture):
            # Check how many mixture components leads to the highest likelihood
            mix_max_likelihood = idx_max([mix[str(n_components)]['likelihood'] for n_components in [1, 2, 3]]) + 1
            max_populated_class = idx_max(mix[str(mix_max_likelihood)]['populations'])
            mean_of_max_populated_class = mix[str(mix_max_likelihood)]['means'][max_populated_class]
            evals.append(mean_of_max_populated_class)

        self.full_evals.append(self.evals_mixture)
        self.evals.append(evals)

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
        summary['evals'] = self.evals
        summary['full_evals'] = self.full_evals
        try:
            summary['true_means'] = [arm.mu if not hasattr(arm, 'gmm') else arm.mus for arm in self.arms]
            summary['true_sds'] = [arm.sigma if not hasattr(arm, 'gmm') else arm.sigmas  for arm in self.arms]
        except:
            summary['true_means'] = None
            summary['true_sds'] = None
        summary['empirical_means'] = get_means(self.history)
        summary['empirical_sds'] = get_sds(self.history)
        if self.mixture_expected:
            summary['evals_mixture'] = self.evals_mixture
        is_every_arm_predicted = all([arm.is_predicted for arm in self.arms])
        if is_every_arm_predicted:
            summary['predictions'] = [arm.prediction for arm in self.arms]
        return summary


