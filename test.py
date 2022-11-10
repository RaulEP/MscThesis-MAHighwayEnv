from pprint import pprint
import gym
import highway_env
import matplotlib.pyplot as pt
from highway_env import utils
from stable_baselines3 import DQN, A2C, PPO
from tqdm.notebook import trange
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import VecMonitor
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


if __name__ == "__main__":

    env = gym.make('ma-highway-v0')

    log_path = 'highway_a2c/'
    A2C_path = os.path.join('Training', 'Saved Models', 'A2C_model')
    best_model_path = os.path.join('Training', 'Saved Models', 'Best_Model')

    #Configuration
    env.reset()

    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=30, verbose=1)
    eval_callback = EvalCallback(env, 
                    eval_freq=1000, 
                    best_model_save_path=best_model_path, #after 10000 steps is going to check and save in this path
                    verbose=1)

    tmp_path = "logger/a2c/"

    # set up logger
    new_logger = configure(tmp_path, ["stdout"])

    ###MODEL TESTING###
    model = A2C.load('Training/Saved Models/best_model', env=env)
    #model = A2C.load('Training/Saved Models/A2C_model', env=env)
    for episode in trange(2, desc="Test episodes"):
        obs, done = env.reset(), False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            env.render('human')
    env.close()
