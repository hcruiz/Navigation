# Navigation in Bananas Collector Environment

This report provides a description of the Deep Q-Learning implementation to solve the navigation and recollection problem of the (Unity) Bananas Environment. 
The implementation is such that, in principle, it is compatible with any Unity environment, although this has not been tested.

## Learning Algorithm

### Deep Q-Learning 
Deep Q-Learning (or DQN) uses a neural network (NN) to represent the action-value tuple q(s,a) for a given state s as an input to the NN. The algorithm adapts the weights of NN to target the (unknown) optimal action-value function _q*(s,a)_ from which we can extract optimal behaviour by choosing the action _a_ that maximizes the value _q*(a,s)_ at a given state _s_. The weight adaptation is done by gradient descent over the expected quadratic loss between the target action value function and the _predicted_ values by the NN. 
Since the target is unknown, we approximate the target by the so-called Temporal-Difference target composed of the reward and the maximized action-value q(s,a) for the following state. Hence, we have an iterative procedure in which we evaluate the Q-values for the sampled states and update the NN using this information. 
In order to stabilize the learning procedure, it is required to fix the maximum Q-value for the target for some update steps. This is called _fixed Q-values_ and it decouples the target from the parameter updates. Notice that this approach requires that we initialize 2 different NNs, one used to sample the actions (the 'actor' network), and one to evaluate the targets (the 'target' network). Hence, to ensure that the actor and the target networks synchronize, we have also to update the target network in the direction of the actor network.
An additional characteristic of the vanilla DQN algorithm is the use of a 'replay buffer' in which the sampled state, action, reward and next state (S,A,R,S') _experience tuples_ are saved. Then, when updating the weights, a mini-batch is sampled from this buffer uniformly to evaluate the gradient and make the update. This _experience replay_ breaks the correlation between the samples used in the update step and stabilizes learning.
#### Pseudo-code
```
 Initialize experience replay buffer with capacity C
 Initialize NN with random weights
 Initialize epsilon
 
 for i_episode = 1,...,N:
     Initialize state
     for t = 1,...,T:
         Get epsilon-greedy action given state
         Get next_state and reward given action
         Check if the episode is done
         Save experience tuple in replay buffer
         Every K steps:
            Update agent's weights
         state = next_state
         if done: 
            break             
     end for
     Decrease epsilon: epsilon = max(min_epsilon, decay*epsilon)
 end for
```

The epsilon-greedy action selection is as follows: with probability epsilon draw uniformly an action, otherwise return the action with maximal action-value.
Notice that, while at each time step the resulting experience tuple is saved to the replay buffer, the gradient descent step is performed only every K steps. The optimizer of the neural network is chosen to be ADAM. 
In addition, every time the actor network is updated, we update the target network by a small quantity _tau_ in the direction of the actor network, e.g. with an exponential moving average update.
The hyper-parameters for this environment can be found in the following table: 

Parameter | Value
---------- | -------
N (# episodes)| 2000
T (max. time) | 1000
epsilon (start) | 1.0
min_epsilon | 0.01
decay | 0.995 
C (buffer size) | 1e5
mini-batch | 64
gamma | 0.99
tau | 1e-3
learning rate | 5e-4  
K (update freq.) | 4   

__Note:__ The algorithm stops if the average score over 100 episodes is larger than +14.

### The neural network architecture
The neural network used as agent for this report is a simple 2-hidden layers neural network with ReLU activation. The hidden layers have 50 neurons each and are fully connected between them, with the input layer (37x50) and the output layer (50x4). The weight initialization of each layer is the default initialization of nn.Linear in PyTorch. The output layer is simply a linear readout layer giving the 4 Q-values, one for each available action.


## Results
![Plot of Rewards](https://github.com/hcruiz/Navigation/blob/master/Score_smoothed_vs_Episode.png)

The plot above shows that the agent learned the correct actions to receive an average reward of +14 over 100 episodes. The condition of 
the environment to be solved was set to a higher average score (+14) than the required. As shown in the figure, this target was met after 500
episodes. In addition it was observed, that the agent can get a reward up to +15 in less than 2000 episodes.


## Future Work

The performance of the vanilla DQN agent with similar parameters used in the LunarLander gym environment was surprisingly efficient. However, the learning profile hints at room for improvement. Here, we describe future changes to the vanilla implementations that will be explored:

* Thorough hyper-parameter search. A systematic grid search could be performed to obtain better parameters than the currently used and to study the behaviour of the agent depending on K, min_epsilon, decay, tau, gamma (from least to most interesting).
* A more systematic exploration of the architecture probably improves performance.
* A simple but interesting modification is to perform few gradient descent updates before updating the target network (as opposed to a single update like now). 
* Implementation of Priority Experience Replay and tuning the hyper-parameters of this step.
* Implementation of the Double Q-Learning procedure is straightforward and a promising modification.
* Implementation of the Duelling Network Architecture could help if the Q-value function has a state-dependent offset and their dependent on the actions ('advantage values') is additive. The improvement of this modification might, however, be more significant if pixel values are used as state observations. 
* Finally, other modifications included in the Rainbow algorithm could help. Each of these modifications (Multi-step bootstrap targets, Distributional DQN and Noisy DQN) should be explored progressively.
