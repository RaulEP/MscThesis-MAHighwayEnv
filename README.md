# highway-env

[![build](https://github.com/eleurent/highway-env/workflows/build/badge.svg)](https://github.com/eleurent/highway-env/actions?query=workflow%3Abuild)
[![Documentation Status](https://readthedocs.org/projects/highway-env/badge/?version=latest)](https://highway-env.readthedocs.io/en/latest/?badge=latest)
[![Downloads](https://img.shields.io/pypi/dm/highway-env)](https://pypi.org/project/highway-env/)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/63847d9328f64fce9c137b03fcafcc27)](https://app.codacy.com/manual/eleurent/highway-env?utm_source=github.com&utm_medium=referral&utm_content=eleurent/highway-env&utm_campaign=Badge_Grade_Dashboard)
[![Coverage](https://codecov.io/gh/eleurent/highway-env/branch/master/graph/badge.svg)](https://codecov.io/gh/eleurent/highway-env)
[![GitHub contributors](https://img.shields.io/github/contributors/eleurent/highway-env)](https://github.com/eleurent/highway-env/graphs/contributors)
[![Environments](https://img.shields.io/github/search/eleurent/highway-env/import%20filename:*_env%20path:highway_env/envs?label=environments)](#the-environments)

A collection of environments for *autonomous driving* and tactical decision-making tasks

<p align="center">
    <img src="https://raw.githubusercontent.com/eleurent/highway-env/master/../gh-media/docs/media/highway-env.gif?raw=true"><br/>
    <em>An episode of one of the environments available in highway-env.</em>
</p>

## The environments

### Highway

```python
env = gym.make("highway-v0")
```

In this task, the ego-vehicle is driving on a multilane highway populated with other vehicles.
The agent's objective is to reach a high speed while avoiding collisions with neighbouring vehicles. Driving on the right side of the road is also rewarded.

<p align="center">
    <img src="https://raw.githubusercontent.com/eleurent/highway-env/master/../gh-media/docs/media/highway.gif?raw=true"><br/>
    <em>The highway-v0 environment.</em>
</p>

A faster variant, `highway-fast-v0` is also available, with a degraded simulation accuracy to improve speed for large-scale training.

## Installation

`pip install highway-env`

## Usage

```python
import gym
import highway_env

env = gym.make("highway-v0")

done = False
while not done:
    action = ... # Your agent code here
    obs, reward, done, info = env.step(action)
    env.render()
```

## Documentation

Read the [documentation online](https://highway-env.readthedocs.io/).

## Citing

If you use the project in your work, please consider citing it with:
```bibtex
@misc{highway-env,
  author = {Leurent, Edouard},
  title = {An Environment for Autonomous Driving Decision-Making},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/eleurent/highway-env}},
}

