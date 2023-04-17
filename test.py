from pprint import pprint
import numpy as np
from tqdm.notebook import trange
import gym
import highway_env
import os 
import matplotlib.pyplot as plt 
from stable_baselines3 import DQN, PPO, A2C
#from sb3_contrib import TRPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from matplotlib import pyplot as pyplot
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#Create Environment
env = gym.make('ma-highway-v0')

#load Model
#model = A2C.load('Training/Saved Models/Best_Model/best_model', env=env)
#model = A2C.load('MscThesis-MAHighwayEnv/Training/Saved Models/BESTTrained_modelA2C16', env=env)
#model = A2C.load('Training/Saved Models/Trained_modelA2C8', env=env)
model = PPO.load('Training/Saved Models/PPO_model-10k-230417-24A', env=env)
#model = TRPO.load('Training/Saved Models/Trained_modelTRPO8', env=env)

env.configure(
    {
        "human_count":0,
        "DLC_config": {
            "count": 12,
            "reward_speed_range": [23, 28], #speed range should be bellow target speed > 28.
            "weights": [1,133,1,1,2,1],
            },
        "MLC_config": {
            "count": 12,
            "reward_speed_range": [19, 23],
            "weights": [1,133,1,1,1,2],
            },
    })

for episode in trange(5, desc="Test episodes"):
    (obs, info), done = env.reset(return_info=True), False
    step = 1
    #while not done:
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        #action = np.array([1,1,1,1,1,1,1,1], dtype=np.int64)
        #action = np.array([4], dtype=np.int64)
        obs, reward, done, info = env.step(action)
        env.render(mode='human')
        step += 1
env.close()
