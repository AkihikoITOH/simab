# /usr/bin/env python
# -*- encoding: utf-8 -*-

import unittest
import os
import sys
from simab.arms.normal import NormalArm

ROUNDS = 100

class TestNormalArm(unittest.TestCase):
    def test_pick(self,):
        arm = NormalArm(0.5, 0.2)
        reward = arm.pick()
        self.assertEqual(isinstance(reward, float), True)

    def test_prediction(self,):
        arm = NormalArm(0.5, 0.2)
        arm.predict(max_rounds=ROUNDS)
        results = []
        for _ in range(ROUNDS):
            results.append(arm.pick())
        self.assertEqual(arm.prediction, results)

        arm = NormalArm(0.5, 0.2)
        arm.predict(prediction=results)
        self.assertEqual(arm.prediction, results)

if __name__ == '__main__':
    unittest.main()

