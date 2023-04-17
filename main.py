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
import tensorboard

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


if __name__ == "__main__":

    env = SubprocVecEnv([
                    lambda: gym.make('ma-highway-v0'),
                    lambda: gym.make('ma-highway-v0'),
                    lambda: gym.make('ma-highway-v0'),
                    lambda: gym.make('ma-highway-v0'),
                    lambda: gym.make('ma-highway-v0'),
                    lambda: gym.make('ma-highway-v0'),
                    lambda: gym.make('ma-highway-v0'),
                    lambda: gym.make('ma-highway-v0'),

    ]) 
    env = VecMonitor(env)

    date = "230417"
    algorithm = "A2C"
    agent_count = 8
    steps = "10k"
    log_path = 'Logs/'
    model_save_path = os.path.join('Training', 'Saved Models', '{}_model-{}-{}-{}A'.format(algorithm, steps, date, agent_count))
    best_model_path = os.path.join('Training', 'Saved Models', '{}_best_model-{}-{}-{}A'.format(algorithm, steps, date, agent_count))

    #Configuration
    env.reset()

    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=30, verbose=1)
    eval_callback = EvalCallback(env,
                    eval_freq=1000,
                    best_model_save_path=best_model_path, #after 10000 steps is going to check and save in this path
                    verbose=1)

    model = A2C('MultiInputPolicy', env, 
                tensorboard_log=log_path,
                verbose=True,
                seed=3,
                normalize_advantage=True)

    model.learn(total_timesteps=int(1000), callback=eval_callback)
    model.save(model_save_path)