# /usr/bin/env python
# -*- encoding: utf-8 -*-

import unittest
from simab.arms.normal import NormalArm
from simab.arms.gmm import GMMArm
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
        algorithm.summary()

    def test_prediction_play(self,):
        arms = []
        for i in range(3, 8):
            arms.append(NormalArm(0.1*float(i), 0.1))
        for arm in arms:
            arm.predict(ROUNDS)
        algorithm = Random(arms)
        for _ in range(ROUNDS):
            algorithm.play()
        self.assertEqual(len(algorithm.history), len(arms))
        self.assertEqual(len(algorithm.history[0]), ROUNDS)
        algorithm.summary()

    def test_mixture(self):
        arms = []
        arms.append(NormalArm(0.2, 0.1))
        arms.append(GMMArm(mus=[0.3, 0.7], sigmas=[0.01, 0.05], weights=[0.4, 0.6]))
        arms.append(GMMArm(mus=[0.1, 0.5, 0.75], sigmas=[0.01, 0.1, 0.02], weights=[0.3, 0.4, 0.3]))
        algorithm = Random(arms)
        for _ in range(ROUNDS):
            algorithm.play()
        self.assertEqual(len(algorithm.history), len(arms))
        self.assertEqual(len(algorithm.history[0]), ROUNDS)
        algorithm.summary()

    def test_mixture_prediction_play(self,):
        arms = []
        arms.append(NormalArm(0.2, 0.1))
        arms.append(GMMArm(mus=[0.3, 0.7], sigmas=[0.01, 0.05], weights=[0.4, 0.6]))
        arms.append(GMMArm(mus=[0.1, 0.5, 0.75], sigmas=[0.01, 0.1, 0.02], weights=[0.3, 0.4, 0.3]))
        algorithm = Random(arms)
        for arm in arms:
            arm.predict(ROUNDS)
        for _ in range(ROUNDS):
            algorithm.play()
        self.assertEqual(len(algorithm.history), len(arms))
        self.assertEqual(len(algorithm.history[0]), ROUNDS)
        algorithm.summary()

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
        algorithm.summary()

    def test_prediction_play(self,):
        arms = []
        for i in range(3, 8):
            arms.append(NormalArm(0.1*float(i), 0.1))
        for arm in arms:
            arm.predict(ROUNDS)
        algorithm = EpsilonGreedy(arms, 0.05)
        for _ in range(ROUNDS):
            algorithm.play()
        self.assertEqual(len(algorithm.history), len(arms))
        self.assertEqual(len(algorithm.history[0]), ROUNDS)
        algorithm.summary()

    def test_mixture(self):
        arms = []
        arms.append(NormalArm(0.2, 0.1))
        arms.append(GMMArm(mus=[0.3, 0.7], sigmas=[0.01, 0.05], weights=[0.4, 0.6]))
        arms.append(GMMArm(mus=[0.1, 0.5, 0.75], sigmas=[0.01, 0.1, 0.02], weights=[0.3, 0.4, 0.3]))
        algorithm = EpsilonGreedy(arms, 0.05, mixture_expected=True)
        for _ in range(ROUNDS):
            algorithm.play()
        self.assertEqual(len(algorithm.history), len(arms))
        self.assertEqual(len(algorithm.history[0]), ROUNDS)
        algorithm.summary()

    def test_mixture_prediction_play(self,):
        arms = []
        arms.append(NormalArm(0.2, 0.1))
        arms.append(GMMArm(mus=[0.3, 0.7], sigmas=[0.01, 0.05], weights=[0.4, 0.6]))
        arms.append(GMMArm(mus=[0.1, 0.5, 0.75], sigmas=[0.01, 0.1, 0.02], weights=[0.3, 0.4, 0.3]))
        algorithm = EpsilonGreedy(arms, 0.05, mixture_expected=True)
        for arm in arms:
            arm.predict(ROUNDS)
        for _ in range(ROUNDS):
            algorithm.play()
        self.assertEqual(len(algorithm.history), len(arms))
        self.assertEqual(len(algorithm.history[0]), ROUNDS)
        algorithm.summary()

class TestEFirst(unittest.TestCase):
    def test_prediction_play(self,):
        arms = []
        for i in range(3, 8):
            arms.append(NormalArm(0.1*float(i), 0.1))
        for arm in arms:
            arm.predict(ROUNDS)
        algorithm = EpsilonFirst(arms, 0.5)
        for i in range(ROUNDS):
            # print '%sth play in epsilon first.' % i
            algorithm.play()
        self.assertEqual(len(algorithm.history), len(arms))
        self.assertEqual(len(algorithm.history[0]), ROUNDS)
        algorithm.summary()

    def test_mixture_prediction_play(self,):
        arms = []
        arms.append(NormalArm(0.2, 0.1))
        arms.append(GMMArm(mus=[0.3, 0.7], sigmas=[0.01, 0.05], weights=[0.4, 0.6]))
        arms.append(GMMArm(mus=[0.1, 0.5, 0.75], sigmas=[0.01, 0.1, 0.02], weights=[0.3, 0.4, 0.3]))
        for arm in arms:
            arm.predict(ROUNDS)
        algorithm = EpsilonFirst(arms, 0.1, mixture_expected=True)
        for i in range(ROUNDS):
            # print '%sth play in epsilon first.' % i
            algorithm.play()
        self.assertEqual(len(algorithm.history), len(arms))
        self.assertEqual(len(algorithm.history[0]), ROUNDS)
        algorithm.summary()

class TestOracle(unittest.TestCase):
    def test_prediction_play(self,):
        arms = []
        for i in range(3, 8):
            arms.append(NormalArm(0.1*float(i), 0.1))
        for arm in arms:
            arm.predict(ROUNDS)
        algorithm = Oracle(arms)
        for _ in range(ROUNDS):
            algorithm.play()
        self.assertEqual(len(algorithm.history), len(arms))
        self.assertEqual(len(algorithm.history[0]), ROUNDS)
        algorithm.summary()

    def test_mixture_prediction_play(self,):
        arms = []
        arms.append(NormalArm(0.2, 0.1))
        arms.append(GMMArm(mus=[0.3, 0.7], sigmas=[0.01, 0.05], weights=[0.4, 0.6]))
        arms.append(GMMArm(mus=[0.1, 0.5, 0.75], sigmas=[0.01, 0.1, 0.02], weights=[0.3, 0.4, 0.3]))
        for arm in arms:
            arm.predict(ROUNDS)
        algorithm = Oracle(arms)
        for i in range(ROUNDS):
            # print '%sth play in epsilon first.' % i
            algorithm.play()
        self.assertEqual(len(algorithm.history), len(arms))
        self.assertEqual(len(algorithm.history[0]), ROUNDS)
        algorithm.summary()

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
        algorithm.summary()

    def test_prediction_play(self,):
        arms = []
        for i in range(3, 8):
            arms.append(NormalArm(0.1*float(i), 0.1))
        for arm in arms:
            arm.predict(ROUNDS)
        algorithm = Softmax(arms, 0.05)
        for _ in range(ROUNDS):
            algorithm.play()
        self.assertEqual(len(algorithm.history), len(arms))
        self.assertEqual(len(algorithm.history[0]), ROUNDS)
        algorithm.summary()

    def test_mixture(self):
        arms = []
        arms.append(NormalArm(0.2, 0.1))
        arms.append(GMMArm(mus=[0.3, 0.7], sigmas=[0.01, 0.05], weights=[0.4, 0.6]))
        arms.append(GMMArm(mus=[0.1, 0.5, 0.75], sigmas=[0.01, 0.1, 0.02], weights=[0.3, 0.4, 0.3]))
        algorithm = Softmax(arms, 0.08, mixture_expected=True)
        for _ in range(ROUNDS):
            algorithm.play()
        self.assertEqual(len(algorithm.history), len(arms))
        self.assertEqual(len(algorithm.history[0]), ROUNDS)
        algorithm.summary()

    def test_mixture_prediction_play(self,):
        arms = []
        arms.append(NormalArm(0.2, 0.1))
        arms.append(GMMArm(mus=[0.3, 0.7], sigmas=[0.01, 0.05], weights=[0.4, 0.6]))
        arms.append(GMMArm(mus=[0.1, 0.5, 0.75], sigmas=[0.01, 0.1, 0.02], weights=[0.3, 0.4, 0.3]))
        for arm in arms:
            arm.predict(ROUNDS)
        algorithm = Softmax(arms, 0.1, mixture_expected=True)
        for i in range(ROUNDS):
            algorithm.play()
        self.assertEqual(len(algorithm.history), len(arms))
        self.assertEqual(len(algorithm.history[0]), ROUNDS)
        algorithm.summary()

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
        algorithm.summary()

    def test_prediction_play(self,):
        arms = []
        for i in range(3, 8):
            arms.append(NormalArm(0.1*float(i), 0.1))
        for arm in arms:
            arm.predict(ROUNDS)
        algorithm = UCB1(arms)
        for _ in range(ROUNDS):
            algorithm.play()
        self.assertEqual(len(algorithm.history), len(arms))
        self.assertEqual(len(algorithm.history[0]), ROUNDS)
        algorithm.summary()

    def test_mixture(self):
        arms = []
        arms.append(NormalArm(0.2, 0.1))
        arms.append(GMMArm(mus=[0.3, 0.7], sigmas=[0.01, 0.05], weights=[0.4, 0.6]))
        arms.append(GMMArm(mus=[0.1, 0.5, 0.75], sigmas=[0.01, 0.1, 0.02], weights=[0.3, 0.4, 0.3]))
        algorithm = UCB1(arms, mixture_expected=True)
        for _ in range(ROUNDS):
            algorithm.play()
        self.assertEqual(len(algorithm.history), len(arms))
        self.assertEqual(len(algorithm.history[0]), ROUNDS)
        algorithm.summary()

    def test_mixture_prediction_play(self,):
        arms = []
        arms.append(NormalArm(0.2, 0.1))
        arms.append(GMMArm(mus=[0.3, 0.7], sigmas=[0.01, 0.05], weights=[0.4, 0.6]))
        arms.append(GMMArm(mus=[0.1, 0.5, 0.75], sigmas=[0.01, 0.1, 0.02], weights=[0.3, 0.4, 0.3]))
        for arm in arms:
            arm.predict(ROUNDS)
        algorithm = UCB1(arms, mixture_expected=True)
        for i in range(ROUNDS):
            # print '%sth play in epsilon first.' % i
            algorithm.play()
        self.assertEqual(len(algorithm.history), len(arms))
        self.assertEqual(len(algorithm.history[0]), ROUNDS)
        algorithm.summary()

class TestSingle(unittest.TestCase):
    def test_play(self,):
        arms = []
        for i in range(3, 8):
            arms.append(NormalArm(0.1*float(i), 0.1))
        algorithm = Single(arms, 3)
        for _ in range(ROUNDS):
            algorithm.play()
        self.assertEqual(len(algorithm.history), len(arms))
        self.assertEqual(len(algorithm.history[0]), ROUNDS)
        algorithm.summary()

    def test_prediction_play(self,):
        arms = []
        for i in range(3, 8):
            arms.append(NormalArm(0.1*float(i), 0.1))
        for arm in arms:
            arm.predict(ROUNDS)
        algorithm = Single(arms, 3)
        for _ in range(ROUNDS):
            algorithm.play()
        self.assertEqual(len(algorithm.history), len(arms))
        self.assertEqual(len(algorithm.history[0]), ROUNDS)
        algorithm.summary()

    def test_mixture(self):
        arms = []
        arms.append(NormalArm(0.2, 0.1))
        arms.append(GMMArm(mus=[0.3, 0.7], sigmas=[0.01, 0.05], weights=[0.4, 0.6]))
        arms.append(GMMArm(mus=[0.1, 0.5, 0.75], sigmas=[0.01, 0.1, 0.02], weights=[0.3, 0.4, 0.3]))
        algorithm = Single(arms, 2)
        for _ in range(ROUNDS):
            algorithm.play()
        self.assertEqual(len(algorithm.history), len(arms))
        self.assertEqual(len(algorithm.history[0]), ROUNDS)
        algorithm.summary()

    def test_mixture_prediction_play(self,):
        arms = []
        arms.append(NormalArm(0.2, 0.1))
        arms.append(GMMArm(mus=[0.3, 0.7], sigmas=[0.01, 0.05], weights=[0.4, 0.6]))
        arms.append(GMMArm(mus=[0.1, 0.5, 0.75], sigmas=[0.01, 0.1, 0.02], weights=[0.3, 0.4, 0.3]))
        for arm in arms:
            arm.predict(ROUNDS)
        algorithm = Single(arms, 1)
        for i in range(ROUNDS):
            algorithm.play()
        self.assertEqual(len(algorithm.history), len(arms))
        self.assertEqual(len(algorithm.history[0]), ROUNDS)
        algorithm.summary()

if __name__ == '__main__':
    unittest.main()

