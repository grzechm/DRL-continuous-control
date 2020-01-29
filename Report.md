#Report

Actor-critic methods are at the intersection of policy-based methods such as reinforce and value-based methods such as DQN. If a deep neural network is used to approximate a value function, the deep reinforcement learning agent is said to be value-based. If the network is used to approximate a policy, the agent is said to be policy-based. 

A value function can be used as a baseline (training target) for a policy-based agent to reduce it's variance.

The Monte-Carlo esimate consists of rolling out an episode in calculating the discounted total reward from the reward sequence. After every episode the network is updated. The more estimates are provided the better the value function will be. Monte-Carlo methods will have high variance because estimates for a state can vary greatly across episodes. The reason for high variance is compounding lots of random events that happened during the course of a single episode. But the methods are unbiased as there is no recursive, local estimation of a state during the learning proceess. So given much enugh data, the estimate shall be accurate.

The temporal difference method relies(?) on estimating the value of the current state using a single reward sample. Agent calculates current value function basing on the value of the current state and the value of the next state. So it basically optimizes the action basing on possible outcomes of the nest states(?). TD methods have low variance because at a single learning step one step of an actor is compunded. There is not much randomness but bootstrapping on the next state estimates adds bias into calculation. The agent will learn faster than in Monte-Carlo methods, will have lower variance, but will have higher bias.



[source: https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf]

###Model architecture

The input to the neural network consist of 37 values. There are 64 nodes for each first and second hidden layer. The final output layer provides 4 values for each valid action. All nodes are followed by a rectified linear unit (ReLU). The network consists only of fully connected layers.

###Hyperparameters

dqn_agent.py file:  
BUFFER_SIZE = 1e5  - size of the replay buffer  
BATCH_SIZE = 64  - number of samples used during learning step  
TAU = 1e-3  - local parameters to target update factor  
LR - 5e-4 - learning rate  
UPDATE_EVERY = 4 - how often to update the network  

Navigation.ipnyb file:  
max_t = 1000  - maksimum number of agent's steps before an episode finishes  
eps_start = 1.0  - initial value of the epsilon paramater  
eps_end = 0.01  - minimal value of the epsilon parameter  
eps_deay = 0.995  - decay rate of the epsilon paramter  

###Plot of rewards 

The environment has been solved (mean score over 100 episodes was higher than 13) in less than 500 episodes.  
![mean score](final_plot_no_PER.png)  

Weights and model are stored in file 'model_weigths.pth'  

###Future improvement

Prioritized Experience Replay (PER):  
Prioritized experience replay relies on the idea that the agent can learn more from some transisions than the others. The more important ones shall be sampled more frequently than the others.  

[source: https://arxiv.org/pdf/1511.05952.pdf]

Double Deep Q-Network (DDQN):  
To avoid overestimation of action values Double DQN has been proposed.  

[source: https://arxiv.org/pdf/1509.06461.pdf]

Dueling DQN:  
With a dueling architecture, the values of each state can be accessed, without learning the effect of each action.

[source: https://arxiv.org/pdf/1511.06581.pdf]
