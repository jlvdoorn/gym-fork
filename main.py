import gym
import time
import numpy as np

class Sim:

    def __init__(self, steps: int =1000):
        self.steps = steps
        self.env = gym.make(
            "LunarLander-v2",
            continuous = False,
            gravity = -9.81,
            enable_wind = False,
            wind_power = 7.5,
            turbulence_power = 0.5,
        )
        observation, info = self.env.reset(seed=42, return_info=True)

    def iterate(self):

        simData = {}

        act = {}
        obs = {}
        rwd = {}
        don = {}
        inf = {}

        episode = 1

        for k in range(self.steps):
            action = self.env.action_space.sample()
            observation, reward, done, info = self.env.step(action)
            self.env.render(mode='human')
            # print('--------------------------------')
            # print('#: '+str(k))
            # print('a: '+str(action))
            # print('t: '+str(observation[4]*180/np.pi))
            # print('x: '+str(observation[0]))
            # print('y: '+str(observation[1]))
            # if info != {}:
            #     print('i: '+str(info))

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

                observation, info = self.env.reset(return_info=True)

            if k == self.steps-1: # finish current episode
                ep = {}
                ep['act'] = act
                ep['obs'] = obs
                ep['rwd'] = rwd
                ep['don'] = don
                ep['inf'] = inf

                simData[episode] = ep

        self.simData = simData

    def oneStep(self):

        action = self.env.action_space.sample()
        observation, reward, done, info = self.env.step(action)
        self.env.render(mode='human')
        
        if done:
            observation, info = self.env.reset(return_info=True)

    def printSimStats(self):
        print('###################')
        print('Simulation done')
        print('# Episodes: '+str(len(self.simData)))
        print('# Iterations: '+str(self.steps))
        print('###################')

        for k in range(1,len(self.simData)+1):
            print('Episode '+str(k))
            print('Iterations: '+str(len(self.simData[k]['act'])))
            print('Min reward: '+str(min(self.simData[k]['rwd'].values())))
            print('Max reward: '+str(max(self.simData[k]['rwd'].values())))
            print('###################')


if __name__ == "__main__":
    falcon9 = Sim(10)
    falcon9.iterate()
    falcon9.printSimStats()
    print('done')
    #env.close()