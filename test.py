from pprint import pprint
from tqdm.notebook import trange
import os
import gym
import highway_env
import matplotlib.pyplot as pt
from highway_env import utils
from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import VecMonitor

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


if __name__ == "__main__":

    
    env = gym.make('ma-highway-v0')

    ###MODEL TESTING###
    model = A2C.load('Training/Saved Models/best_model', env=env)
    for episode in trange(1, desc="Test episodes"):
        obs, done = env.reset(), False
        print("episode " + str(episode + 1))
        step = 1
        #while not done:
        for i in range(3):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            pprint(info["speed_metrics"])
            pprint(info["position_metrics"])
            #env.render('human')
            step += 1
    env.close()
