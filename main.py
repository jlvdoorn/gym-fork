import gym
import time
import numpy as np

env = gym.make(
    "LunarLander-v2",
    continuous = True,
    gravity = -9.81,
    enable_wind = False,
    wind_power = 7.5,
    turbulence_power = 0.5,
)
observation, info = env.reset(seed=42, return_info=True)

def iterate(steps=100000):

    for _ in range(steps):
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        env.render(mode='human')
        print('t: '+str(observation[4]*180/np.pi))
        print('x: '+str(observation[0]))
        print('y: '+str(observation[1]))
        if info != {}:
            print('i: '+str(info))

        if done:
            observation, info = env.reset(return_info=True)


def oneStep():

    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    env.render(mode='human')
    
    if done:
        observation, info = env.reset(return_info=True)

if __name__ == "__main__":
    iterate(500)
    #env.close()