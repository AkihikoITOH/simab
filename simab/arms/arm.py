#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import uuid

class Arm(object):
    def __init__(self, label=None):
        self.label = label or str(uuid.uuid4())
        self.prediction = None
        self.count = 0

    def pick(self):
        self.count += 1
        if self.prediction is not None:
            reward = self.prediction[self.count]
        else:
            reward = None
        return None

    def predict(self, max_rounds):
        """ Set designated number of values as a prediction.
        This method is used if you want to compare the performance of each algorithm to oracle.

        :param max_rounds: maximum number of rounds to predict
        """
        self.prediction = [self.pick() for _ in range(max_rounds)]

