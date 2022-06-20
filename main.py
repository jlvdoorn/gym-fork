from textwrap import indent
from xml.dom.xmlbuilder import DocumentLS
import gym
import time
import numpy as np

import random
from keras import Sequential
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras.activations import relu, linear

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

    def trainDQN(self, episode):
        loss = []
        agent = DQN(self.env.action_space.n, self.env.observation_space.shape[0])

        for e in range(episode):
            state = self.env.reset()
            state = np.reshape(state, (1, 8))
            score = 0
            max_steps = 3000

            for i in range(max_steps):
                action = agent.act(state)
                self.env.render(mode=None)

                next_state, reward, done, _ = self.env.step(action)
                score += reward
                next_state = np.reshape(next_state, (1, 8))

                agent.remember(state, action, reward, next_state, done)
                state = next_state

                agent.replay()
                
                if done:
                    print('episode: {}/{}, score: {}'.format(e, episode, score))
                    break
            loss.append(score)

            # Average score of last 100 episodes
            is_solved = np.mean(loss[-100:])
            if is_solved > 200:
                print('\n Task Completed! \n')
                break
            print('Average over last 100 episodes: {0:.2f} \n'.format(is_solved))
        self.loss = loss
        return self.loss
    
    def plotGraph(self):
        plt.plot([i+1 for i in range(0, len(self.loss), 2)], self.loss[::2])
        plt.show()


class DQN:
    def __init__(self, action_space, state_space):
        self.action_space = action_space # action space
        self.state_space = state_space # state/observation space
        self.epsilon = 1.0
        self.gamma   = 0.99
        self.batch_size = 64
        self.epsilon_min = 0.01
        self.lr = 0.001
        self.epsilon_decay = 0.996
        self.memory = deque(maxlen=1000000)
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(150, input_dim=self.state_space, activation=relu))
        model.add(Dense(120, activation=relu))
        model.add(Dense(self.action_space, activation=linear))
        model.compile(loss='mse', optimizer=Adam(lr=self.lr))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state,action,reward,next_state,done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states      = np.array([i[0] for i in minibatch])
        actions     = np.array([i[1] for i in minibatch])
        rewards     = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones       = np.array([i[4] for i in minibatch])

        states      = np.squeeze(states)
        next_states = np.squeeze(next_states)

        targets      = rewards + self.gamma*(np.amax(self.model.predict_on_batch(next_states), axis=1))*(1-dones)
        targets_full = self.model.predict_on_batch(states)
        ind 	     = np.array([i for i in range(self.batch_size)])
        targets_full[[ind], [actions]] = targets

        self.model.fit(states, targets_full, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

if __name__ == "__main__":
    falcon9 = Sim(1000)
    falcon9.trainDQN(200) # 5 episodes take approx 1 minute
    falcon9.plotGraph()
    print(falcon9.loss)
