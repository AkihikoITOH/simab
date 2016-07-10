# simab - Simple Multi-Armed Bandit Simulator

[![PyPI version](https://badge.fury.io/py/simab.svg)](https://badge.fury.io/py/simab)
[![Build Status](https://travis-ci.org/AkihikoITOH/simab.svg?branch=master)](https://travis-ci.org/AkihikoITOH/simab)
[![Test Coverage](https://codeclimate.com/github/AkihikoITOH/simab/badges/coverage.svg)](https://codeclimate.com/github/AkihikoITOH/simab/coverage)
[![Code Climate](https://codeclimate.com/github/AkihikoITOH/simab/badges/gpa.svg)](https://codeclimate.com/github/AkihikoITOH/simab)

## Algorithms

- [x] Epsilon First
- [x] Epsilon Greedy
- [x] Softmax
- [x] UCB1
- [x] Random: randomly choose an arm
- [x] Oracle: always choose the best arm
- [x] Single: keep choosing a specific arm

## Arms

- [x] Normal Distribution
- [ ] Gaussian Mixture Model

## Installation

```python
$ pip install simab
```

## Usage

### Basic

```python
from simab.arms.normal import NormalArm
from simab.algorithms.epsilon_first import EpsilonFirst
from simab.algorithms.epsilon_greedy import EpsilonGreedy
from simab.algorithms.oracle import Oracle
from simab.algorithms.random_choice import Random
from simab.algorithms.single_arm import Single
from simab.algorithms.softmax import Softmax
from simab.algorithms.ucb1 import UCB1

ROUNDS = 1000

# Generate five ND-arms.
arms = [NormalArm(0.1*float(i), 0.1) for i in range(3, 8)]

# Generate an agent.
algorithm = Softmax(arms, 0.1)
# algorithm = EpsilonFirst(arms, 0.1, rounds=ROUNDS)
# algorithm = EpsilonGreedy(arms, 0.1)
# algorithm = UCB1(arms)
# algorithm = Single(arms, 2)
# algorithm = Random(arms)

for _ in range(ROUNDS):
    algorithm.play()

# Get the summary.
summary = algorithm.summary()
print summary['algorithm']
print summary['tau']
print summary['plays']
print summary['total_reward']
print summary['true_means']
print summary['true_sds']
print summary['empirical_means']
print summary['empirical_sds']
print summary['history']
```

### Predicted(Simulated) MAB

You need to generate reward from each arm at each round at first if you conduct simulation with `Oracle`.

```python
ROUNDS = 1000
arms = [NormalArm(0.1*float(i), 0.1) for i in range(3, 8)]

# Generate predictions
for arm in arms:
    arm.predict(1000)

algorithm = Oracle(arms)

for _ in range(ROUNDS):
    algorithm.play()
```

If you want to simulate algorithms and compare them to `Oracle`, it's better use the same arms yielding exactly the same reward for each round. In such situations, you can simply `reset()` arms to the initial states while they keep predictions.

```python
ROUNDS = 1000
arms = [NormalArm(0.1*float(i), 0.1) for i in range(3, 8)]
for arm in arms:
    arm.predict(1000)

for algorithm in [Softmax(arms, 0.1), EpsilonFirst(arms, 0.1, rounds=ROUNDS), EpsilonGreedy(arms, 0.1), UCB1(arms), Single(arms, 2), Random(arms)]:
    for _ in range(ROUNDS):
        algorithm.play()
    for arm in arms:
        arm.reset()
```

