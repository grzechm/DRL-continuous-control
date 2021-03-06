# DRL-continuous-control
Continous control concept in deep reinforcement learning.

Agent taking random actions  
![agent performance visualisation](reacher_random.gif)  


Trained agent  
![agent performance visualisation](reacher_actor.gif)


### Introduction  

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of an agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

 In particular, the agents must get an average score of +30 (over 100 consecutive episodes, and over all agents). Specifically,
 
 * After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 20 (potentially different) scores. We then take the average of these 20 scores.
 
 * This yields an average score for each episode (where the average is over all 20 agents).


### Dependencies (OS: Ubuntu 18.04)  

Install Anaconda - https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html  
Run install.sh file to add required components.

Clone the repository, in conda 'drlnd' environment start jupyter notebook by typing in terminal `jupyter-notebook`, switch kernel to 'drlnd'.


Download built environment - https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip

### Training the agent
To train the agent use ContinousControl.ipynb file  
Type in terminal:  
	jupyter-notebook ContinousControl.ipnyb

To visualise use ReacherVisualisation.ipnyb
