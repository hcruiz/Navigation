# Navigation
This repository contains my solution to the Navigation project from the Udacity Nanodegree "Deep Reinforcement Learning". 
The task is to solve a Bananas environment similar to the [Banana Collector](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#banana-collector) 
of the Unity game engine, but the code should work for any Unity environment.

## Project Details

The Bananas environment consists of a square arena with blue and yellow bananas. The agent must learn to collect only yellow bananas and avoid the blue ones. The environment provides a reward of +1 for the yellow and -1 for the blue bananas.
The agent collects bananas while navigating, so in order to collect as many bananas as possible before the episode ends, it must act by choosing 4 possible discrete actions: go forward/backwards or turn left/right. These are coded as the indices of an array in python:

* `0` : move forward
* `1` : move backwards
* `2` : turn left
* `3` : turn right

The information that the agent receives to make a decision is a state vector of 37 dimensions containing the agent's velocity (in the plane) and 7 'ray-based' perceptual information. For more details on these ray-based perception in the Udacity Banana environment, please read [here](https://github.com/Unity-Technologies/ml-agents/issues/1134).
Interestingly, we do not need to know the details/significance of the information provided by the environment. The RL agent should figure it out on its own!

The environment is considered solved, if the agent has an average score of +13 over 100 consecutive episodes.

## Getting Started

First, install Anaconda (python 3) and clone/download this repository (from terminal use the `git clone` command). To install all the required
packages needed to run this task you can create an envarionment using the .yml file in this repository. Just run on your terminal

`conda env create -f environment.yml`

This environment is based on the environment provided by Udacity for this project, with the addition of the specific [PyTorch](https://pytorch.org/) version that I required.
To activate the environment run `conda activate drlnd` and verify that the environment is installed correctly using `conda list`.

__NOTE:__ The torch version in this environment assumes Windows 10 and __no CUDA__ installation. If you want to run the neural networks using 
CUDA, please make sure you install the proper PyTorch version found [here](https://pytorch.org/get-started/locally/). 

Finally, you have to download the Bananas Unity environment. There are different versions depending on your operating system, so please make sure you have the correct version of the environment. The files of the environment must be placed in the repository directory or, if placed somewhere else, the initialization of the environment in the dqn_training.py file must contain the path to the environment.

## Instructions

The code is structured as follows. There are three files: model.py, dqn_agent.py and dqn_training.py. The model file contains the neural network mapping the state to the action-value vector, the aqn_agent file contains a class Agent initializing the netowrk, performing the action selection and the update. In addition, it contains a class for the replay buffer. The dqn_training file contains a function train_dqn that implements training with default hyper-parameter values. 

To train the agent, go to the repo directory in your terminal and run `python qdn_training.py`. This will open a window showing the agent in the environment while training. Once finalized, it will show the averaged score per episode and an example episode of the agent after training. 

To train the agent in another environment you need to initialized it accordingly in the _main_ section of dqn_training.py.

NOTE: The code is based on examples provided, modified and extended during the Udacity DRL Nanodegree.
