from pprint import pprint
import gym
import highway_env
import matplotlib.pyplot as pt
from highway_env import utils
from stable_baselines3 import DQN, A2C, PPO
from tqdm.notebook import trange
import os

env = gym.make('ma-highway-v0')
env.configure(  {
                "speed_limit": 20,
                "vehicles_density": 1,
                "ego_spacing": 1.5,
                "road_length":2000, 
                "simulation_frequency":60, 
                "duration":40,
                "screen_width": 1000,
                "DLC_config": {
                    "count": 3,
                    "reward_speed_range": [20, 40],
                    "weights": [10,5,1,1],
                        },
                "MLC_config": {
                    "count":5 ,
                    "reward_speed_range": [20, 30],
                    "weights": [2,10,1,1]
                        } 
                })

A2C_path = os.path.join('Training', 'Saved Models', 'A2C_model')

"""
#MODEL CREATION
TRAIN = True
model = A2C('MultiInputPolicy', env,
            policy_kwargs=dict(net_arch=[256, 256]),
            learning_rate=5e-4,
            gamma=0.8,
            verbose=1,
            tensorboard_log="highway_a2c/",
            normalize_advantage=True)

#TRAINING
if TRAIN:
    model.learn(total_timesteps=int(1000))
    model.save(A2C_path)
    #del model
"""

"""
###MODEL TESTING###
model = PPO.load('Training/Saved Models/A2C_model', env=env)
for episode in trange(3, desc="Test episodes"):
    obs, done = env.reset(), False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        print(info)
        env.render('human')
env.close()
"""

###TESTING CODE#####
for trials in range(50):
    terminated = False
    obs = env.reset()
    while not terminated:
        action = env.action_space.sample()
        obs, reward, terminated, info = env.step(action)
        env.render('human')
        pprint([obs, reward, terminated])
env.close()

