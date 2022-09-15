from pprint import pprint
import gym
import highway_env
import matplotlib.pyplot as pt

env = gym.make('ma-highway-v0')
print("llegue aqui")
env.reset()

for i in range(5):
    env.configure({'simulation_frequency':15, 'show_trajectories':True, 'initial_lane_id':1, 'duration':10, 'lanes_count':3})
    action = env.action_space.sample()
    obs, reward, terminated, info = env.step(action)
    #everything = env.step(action)
    env.render(mode='human')
    pprint([obs, reward, terminated, info])
    #pprint(everything)
"""x = env.render('rgb_array')
pt.plot(x)
"""

"""
TRAIN = True

if __name__ == '__main__':
    # Create the environment
    env = gym.make("highway-fast-v0")
    obs = env.reset()

    # Create the model
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
                tensorboard_log="highway_dqn/")

    # Train the model
    if TRAIN:
        model.learn(total_timesteps=int(2e4))
        model.save("highway_dqn/model")
        del model

    # Run the trained model and record video
    model = DQN.load("highway_dqn/model", env=env)
    env = RecordVideo(env, video_folder="racetrack_ppo/videos", episode_trigger=lambda e: True)
    env.unwrapped.set_record_video_wrapper(env)
    env.configure({"simulation_frequency": 15})  # Higher FPS for rendering

    for videos in range(10):
        done = False
        obs = env.reset()
        while not done:
            # Predict
            action, _states = model.predict(obs, deterministic=True)
            # Get reward
            obs, reward, done, info = env.step(action)
            # Render
            env.render()
    env.close()
"""