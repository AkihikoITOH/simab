#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import uuid

class Arm(object):
    """ Abstract Arm class.

    :param label: labels of the arm. Defaulted to an UUID.
    """
    def __init__(self, label=None):
        self.label = label or str(uuid.uuid4())
        self.prediction = None
        self.is_predicted = False
        self.count = 0

    def reset(self):
        self.count = 0

    def pick(self, dry=False):
        """ Pick a reward from prediction only if prediction exists.
        """
        if self.is_predicted:
            reward = self.prediction[self.count]
        else:
            reward = None
        if not dry:
            self.count += 1
        return reward

    def predict(self, max_rounds=1000, prediction=[]):
        """ Set designated number of values as a prediction.
        This method is used if you want to compare the performance of each algorithm to oracle.

        :param max_rounds: maximum number of rounds to predict
        :param prediction: float list of prediction
        """
        if prediction and len(prediction)>0:
            self.prediction = prediction
        else:
            self.prediction = [self.pick(dry=True) for _ in range(max_rounds)]
        self.is_predicted = True


