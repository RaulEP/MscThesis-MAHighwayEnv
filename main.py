from pprint import pprint
import gym
import highway_env
import matplotlib.pyplot as pt
from highway_env import utils
from stable_baselines3 import DQN
from tqdm.notebook import trange
import os

env = gym.make('ma-highway-v0')
DQN_path = os.path.join('Training', 'Saved Models', 'DQN_model')

env.configure({"vehicles_count": 25, "vehicles_density": 1.3, "simulation_frequency":15, "duration":40, "road_length":2000})
env.reset()


#MODEL CREATION
TRAIN = True
model = DQN('MlpPolicy', env,
            policy_kwargs=dict(net_arch=[256, 256]),
            learning_rate=5e-4,
            buffer_size=15000,
            learning_starts=200,
            batch_size=32,
            gamma=0.8,
            train_freq=1,
            gradient_steps=1,
            target_update_interval=50,
            verbose=1,
            tensorboard_log="highway_dqn/",
            exploration_fraction=0.7)

#TRAINING
if TRAIN:
    model.learn(total_timesteps=int(3e4))
    model.save(DQN_path)
    #del model

"""
###MODEL TESTING###
model = DQN.load('highway_dqn/model', env=env)
for episode in trange(10, desc="Test episodes"):
    obs, done = env.reset(), False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(int(action))
        env.render('human')
env.close()



###TESTING CODE#####
for trials in range(1):
    env.configure({'show_trajectories':True, 'initial_lane_id':2, 'lanes_count':3, "vehicles_count": 1, })
    terminated = False
    obs = env.reset()
    while not terminated:
        action = env.action_space.sample()
        obs, reward, terminated, info = env.step(action)
        env.render(mode='human')
        pprint([obs, reward, terminated, info])
env.close()
"""