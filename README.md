[![CI](https://github.com/jlvdoorn/gym-fork/actions/workflows/main.yml/badge.svg)](https://github.com/jlvdoorn/gym-fork/actions/workflows/main.yml)

## Gym-Fork

This is a fork of the OpenAI Gym library, for more info see [https://www.gymlibrary.ml/](https://www.gymlibrary.ml/).

## SpaceX Falcon-9 Booster Rocket

In this fork, the original lunar lander environment is adapted such that it models the Falcon-9 Booster Rocket of SpaceX. With this enivronment a machine learning (specificlly: reinforcement learning using neural network) model is applied in order to optimize the rocket to land itself. That was the whole idea from SpaceX behind this rocket. 

## Usage

By running ```main.py``` the simulation will start. In the beginning the rocket will try to land itself but fail and after a given amount of episodes (tries) the rocket will be able to land itself perfectly in the center of the designated area. Afterwards, it will produce a graph of the total reward obtained for each of the episodes, there the learning curve will be visible clearly.

## Results
No training
![no_training](https://media0.giphy.com/media/bAM0xaTvbn9OZl3bzf/giphy.gif?cid=790b761150b61c1125b475c5a37be0164ff2641cbc911f29&rid=giphy.gif&ct=g)

Mid training
![mid_training](https://media1.giphy.com/media/uf9QucTxSUJ8dbT17O/giphy.gif?cid=790b7611ebc8369589f220be1947b76af1711ec44e0329df&rid=giphy.gif&ct=g)

After training (100 episodes of max 100k steps)
![after_training](https://media0.giphy.com/media/E5pLGIYpib2Etwhkhg/giphy.gif?cid=790b7611d78b7b504cec730df5c4056c90efb8e0d3075ef6&rid=giphy.gif&ct=g)
