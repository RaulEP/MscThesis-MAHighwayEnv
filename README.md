# Korea University MsC Thesis by Raul Pena

In this repository I created a new gym environment called MAHighway-env based on the original "HighwayEnv" library by Edouard Leurent [documentation online](https://highway-env.readthedocs.io/). 

In this environment I integrated two vehicle category types MLC Vehicle and DLC Vehicle, each vehicle has different lane changing objectives and this enviroment provides a training bed for multi-agent lane changing decision for each type of vehicle at each time step. 

## Usage

```python
import gym
import highway_env

env = gym.make("ma-highway-v0")

done = False
while not done:
    action = ... # Your agent code here
    obs, reward, done, info = env.step(action)
    env.render()
```

## Original REPO Documentation

Read the [documentation online](https://highway-env.readthedocs.io/).

