# /usr/bin/env python
# -*- encoding: utf-8 -*-

import unittest
import os
import sys
from simab.arms.normal import NormalArm
from simab.algorithms.epsilon_first import EpsilonFirst
from simab.algorithms.epsilon_greedy import EpsilonGreedy
from simab.algorithms.oracle import Oracle
from simab.algorithms.random_choice import Random
from simab.algorithms.single_arm import Single
from simab.algorithms.softmax import Softmax
from simab.algorithms.ucb1 import UCB1

ROUNDS=1000

class TestRandom(unittest.TestCase):
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

class TestEGreedy(unittest.TestCase):
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

class TestEFirst(unittest.TestCase):
    def test_prediction_play(self,):
        arms = []
        for i in range(3, 8):
            arms.append(NormalArm(0.1*float(i), 0.1))
        for arm in arms:
            arm.predict(1000)
        algorithm = EpsilonFirst(arms, 0.05)
        for _ in range(ROUNDS):
            algorithm.play()
        self.assertEqual(len(algorithm.history), len(arms))
        self.assertEqual(len(algorithm.history[0]), ROUNDS)

class TestOracle(unittest.TestCase):
    def test_prediction_play(self,):
        arms = []
        for i in range(3, 8):
            arms.append(NormalArm(0.1*float(i), 0.1))
        for arm in arms:
            arm.predict(1000)
        algorithm = Oracle(arms)
        for _ in range(ROUNDS):
            algorithm.play()
        self.assertEqual(len(algorithm.history), len(arms))
        self.assertEqual(len(algorithm.history[0]), ROUNDS)

class TestSoftmax(unittest.TestCase):
    def test_play(self,):
        arms = []
        for i in range(3, 8):
            arms.append(NormalArm(0.1*float(i), 0.1))
        algorithm = Softmax(arms, 0.05)
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
        algorithm = Softmax(arms, 0.05)
        for _ in range(ROUNDS):
            algorithm.play()
        self.assertEqual(len(algorithm.history), len(arms))
        self.assertEqual(len(algorithm.history[0]), ROUNDS)

class TestUCB1(unittest.TestCase):
    def test_play(self,):
        arms = []
        for i in range(3, 8):
            arms.append(NormalArm(0.1*float(i), 0.1))
        algorithm = UCB1(arms)
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
        algorithm = UCB1(arms)
        for _ in range(ROUNDS):
            algorithm.play()
        self.assertEqual(len(algorithm.history), len(arms))
        self.assertEqual(len(algorithm.history[0]), ROUNDS)

class TestSingle(unittest.TestCase):
    def test_play(self,):
        arms = []
        for i in range(3, 8):
            arms.append(NormalArm(0.1*float(i), 0.1))
        algorithm = Single(arms, 5)
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
        algorithm = Single(arms)
        for _ in range(ROUNDS):
            algorithm.play()
        self.assertEqual(len(algorithm.history), len(arms))
        self.assertEqual(len(algorithm.history[0]), ROUNDS)

if __name__ == '__main__':
    unittest.main()

