# DRLND Collaboration & Competition
![tennis-gif](https://github.com/gimaik/DRLND/P3-Collaboration-Competition/blob/master/tennis.gif)

      
# Solution for the Unity Tennis Environment

## Environment Description

For this project, two agents control rackets to bounce a ball over a net in the Unity environment. The goal of each agent is to keep the ball in play. 

### Reward Space
If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. 

### Observation Space
The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. 

### Action Space
Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

### Solution Criteria
The task is episodic, and in order to solve the environment, the agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents).  
* *After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent.*
* *This yields 2 (potentially different) scores. The maximum score of the agent is then the score for the episode. The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.*


## Getting Started
1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:

- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zi)
            
2. In order to train the model or inference the computed weights, the following python packages need to be installed:
* *pytorch*
* *unityagents*
* *numpy*
* *matplotlib* 

## Instructions

Follow the instructions in `Tennis.ipynb` to get started with training your own agent!  





