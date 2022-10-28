from pprint import pprint
import gym
import highway_env
import matplotlib.pyplot as pt
from highway_env import utils
from stable_baselines3 import DQN, A2C, PPO
from tqdm.notebook import trange
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.logger import configure

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

env = gym.make('ma-highway-v0', )


log_path = 'highway_a2c/'
A2C_path = os.path.join('Training', 'Saved Models', 'A2C_model')
best_model_path = os.path.join('Training', 'Saved Models', 'Best_Model')

#Configuration
env.configure( {
                "lanes_count": 3,
                "speed_limit": 33,
                "vehicles_density": 1,
                "normalization_range": [-51, 14],
                "ego_spacing": 1,
                "road_length":10000, 
                "screen_width": 2500, 
                "simulation_frequency":15, 
                "duration":40,
                "normalize_reward": False,
                "DLC_config": {
                    "count": 5,
                    "reward_speed_range": [23, 28], #speed range should be bellow target speed > 28.
                    "weights": [3,50,1,1,1],
                        },
                "MLC_config": {
                    "count":5 ,
                    "reward_speed_range": [19, 23],
                    "weights": [3,50,1,1,1],
                        },
                })
env.reset()


stop_callback = StopTrainingOnRewardThreshold(reward_threshold=30, verbose=1)
eval_callback = EvalCallback(env, 
                eval_freq=1000, 
                best_model_save_path=best_model_path, #after 10000 steps is going to check and save in this path
                verbose=1)

tmp_path = "logger/a2c/"

# set up logger
new_logger = configure(tmp_path, ["stdout"])

model = A2C('MultiInputPolicy', env, 
             tensorboard_log="highway_a2c/2",
             verbose=True,
             seed=3,
            normalize_advantage=True)

# Set new logger
model.set_logger(new_logger)
model.learn(total_timesteps=int(20000), callback=eval_callback)
model.save(A2C_path)

"""
###MODEL TESTING###
model = A2C.load('Training/Saved Models/A2C_model', env=env)
for episode in trange(3, desc="Test episodes"):
    obs, done = env.reset(), False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        print(info)
        env.render('human')
env.close()


###TESTING CODE#####
for trials in range(10):
    terminated = False
    obs = env.reset()
    while not terminated:
        action = env.action_space.sample()
        obs, reward, terminated, info = env.step(action)
        env.render('human')
        pprint([info])
env.close()

"""