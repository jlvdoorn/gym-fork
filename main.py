import gym
env = gym.make(
    "LunarLander-v2",
    continuous = True,
    gravity = -9.81,
    enable_wind = False,
    wind_power = 7.5,
    turbulence_power = 0.5,
)
observation, info = env.reset(seed=42, return_info=True)

for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    env.render(mode='human')

    if done:
        observation, info = env.reset(return_info=True)
env.close()