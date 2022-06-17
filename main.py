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

def iterate(steps=1000):

    simData = {}

    act = {}
    obs = {}
    rwd = {}
    don = {}
    inf = {}

    episode = 1

    for k in range(steps):
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        env.render(mode='human')
        print('--------------------------------')
        print('#: '+str(k))
        print('a: '+str(action))
        print('t: '+str(observation[4]*180/np.pi))
        print('x: '+str(observation[0]))
        print('y: '+str(observation[1]))
        if info != {}:
            print('i: '+str(info))

        act[k] = action
        obs[k] = observation
        rwd[k] = reward
        don[k] = done
        inf[k] = info

        if done:
            ep = {}
            ep['act'] = act
            ep['obs'] = obs
            ep['rwd'] = rwd
            ep['don'] = don
            ep['inf'] = inf

            simData[episode] = ep
            
            episode = episode + 1

            act = {}
            obs = {}
            rwd = {}
            don = {}
            inf = {}

            observation, info = env.reset(return_info=True)

        if k == steps-1: # finish current episode
            ep = {}
            ep['act'] = act
            ep['obs'] = obs
            ep['rwd'] = rwd
            ep['don'] = don
            ep['inf'] = inf

            simData[episode] = ep

    return simData

def oneStep():

    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    env.render(mode='human')
    
    if done:
        observation, info = env.reset(return_info=True)

def printSimStats(simData):
    print('Simulation done')
    print('# Episodes: '+str(len(simData)))

    for k in range(1,len(simData)+1):
        print('Episode '+str(k))
        print(str(len(simData[k]['act']))+' iterations')
        print(str(min(simData[k]['rwd']))+' min reward')
        print(str(max(simData[k]['rwd']))+' max reward')


if __name__ == "__main__":
    simData = iterate(500)
    printSimStats(simData)
    print('done')
    #env.close()