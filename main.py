from pprint import pprint
import gym
import highway_env
import matplotlib.pyplot as pt
from highway_env import utils
from stable_baselines3 import DQN, A2C
from tqdm.notebook import trange
import os

env = gym.make('ma-highway-v0')
DQN_path = os.path.join('Training', 'Saved Models', 'DQN_model')

env.configure({"vehicles_count": 17, "vehicles_density": 2, "simulation_frequency":30, "duration":40, "road_length":2000,  "controlled_vehicles": 8,})
env.reset()

"""
#MODEL CREATION
TRAIN = True
model = A2C('MlpPolicy', env,
            policy_kwargs=dict(net_arch=[256, 256]),
            learning_rate=5e-4,
            gamma=0.8,
            verbose=1,
            tensorboard_log="highway_dqn/",
            normalize_advantage=True)

#TRAINING
if TRAIN:
    model.learn(total_timesteps=int(3e4))
    model.save(DQN_path)
    #del model


###MODEL TESTING###
model = DQN.load('Training/Saved Models/DQN_model', env=env)
for episode in trange(3, desc="Test episodes"):
    obs, done = env.reset(), False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(int(action))
        print(info)
        env.render('human')
env.close()
"""

###TESTING CODE#####
for trials in range(10):
    terminated = False
    obs = env.reset()
    while not terminated:
        action = env.action_space.sample()
        obs, reward, terminated, info = env.step(action)
        env.render(mode='human')
        pprint([obs, reward, terminated, info])
env.close()
