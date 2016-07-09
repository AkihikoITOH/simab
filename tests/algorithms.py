# /usr/bin/env python
# -*- encoding: utf-8 -*-

import unittest
import os
import sys
from simab.arms.normal import NormalArm
from simab.algorithms.random_choice import Random
from simab.algorithms.epsilon_greedy import EpsilonGreedy

ROUNDS=1000

class TestRandomArm(unittest.TestCase):
    def test_play(self,):
        arms = []
        for i in range(3, 8):
            arms.append(NormalArm(0.1*float(i), 0.1))
        algorithm = Random(arms)
        for _ in range(ROUNDS):
            algorithm.play()
        self.assertEqual(len(algorithm.history), len(arms))
        self.assertEqual(len(algorithm.history[0]), ROUNDS)

    def test_prediction_play(self,):
        arms = []
        for i in range(3, 8):
            arms.append(NormalArm(0.1*float(i), 0.1))
        for arm in arms:
            arm.predict(1000)
        algorithm = Random(arms)
        for _ in range(ROUNDS):
            algorithm.play()
        self.assertEqual(len(algorithm.history), len(arms))
        self.assertEqual(len(algorithm.history[0]), ROUNDS)

class TestEGreedyArm(unittest.TestCase):
    def test_play(self,):
        arms = []
        for i in range(3, 8):
            arms.append(NormalArm(0.1*float(i), 0.1))
        algorithm = EpsilonGreedy(arms, 0.05)
        for _ in range(ROUNDS):
            algorithm.play()
        self.assertEqual(len(algorithm.history), len(arms))
        self.assertEqual(len(algorithm.history[0]), ROUNDS)

    def test_prediction_play(self,):
        arms = []
        for i in range(3, 8):
            arms.append(NormalArm(0.1*float(i), 0.1))
        for arm in arms:
            arm.predict(1000)
        algorithm = EpsilonGreedy(arms, 0.05)
        for _ in range(ROUNDS):
            algorithm.play()
        self.assertEqual(len(algorithm.history), len(arms))
        self.assertEqual(len(algorithm.history[0]), ROUNDS)

if __name__ == '__main__':
    unittest.main()

