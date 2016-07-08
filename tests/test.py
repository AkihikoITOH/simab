# /usr/bin/env python
# -*- encoding: utf-8 -*-

import unittest
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), '../simab'))
from arms.normal import NormalArm

class TestNormalArm(unittest.TestCase):
    def test_pick(self,):
        arm = NormalArm(0.5, 0.2)
        arm.pick()
        self.assertEqual(arm.count, 1)
        arm.pick()
        self.assertEqual(arm.count, 2)

    def test_prediction(self,):
        arm = NormalArm(0.5, 0.2)
        arm.predict(100)
        self.assertEqual(len(arm.prediction), 100)

if __name__ == '__main__':
    unittest.main()

