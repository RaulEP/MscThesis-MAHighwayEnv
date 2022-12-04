# Korea University MsC Thesis by Raul Pena

In this repository I created a new gym environment called MAHighway-env based on the original "HighwayEnv" library by Edouard Leurent [documentation online](https://highway-env.readthedocs.io/). 

In this environment I integrated two vehicle category types MLC Vehicle and DLC Vehicle, each vehicle has different lane changing objectives and this enviroment provides a training bed for multi-agent lane changing decision for each type of vehicle at each time step. 

In this library we propose a cooperative lane changing model that considers both MLC vehicle and DLC vehicle categories for the cooperative lane changing decisions. 
As seen in [Msc Thesis Link Here]

Vehicle Type Description:

-DLC Vehicle is defined as a Connected Autonomous Vehicle (CAV) with discretionary lane changing objectives. This vehicle main objective is to maximize its average travelling speed by reaching its target speed. DLC Vehicle target speed is 28 m/s or more, and the vehicle is represented by the red color.

-MLC Vehicle: is defined as a Connected Autonomous Vehicle (CAV) with mandatory lane changing objectives. This vehicle main objective is to arrive to its destination by reaching its target lane. MLC Vehicle target lane is the bottom-most lane, and the vehicle is represented by the blue color.

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

