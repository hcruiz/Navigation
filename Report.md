# Navigation in Bananas Collector Environment

This report provides a description of the Deep Q-Learning implementation to solve the navigation and recollection problem of the (Unity) Bananas Environment. 
The implementation is such that, in principle, it is compatible with any Unity environment, although this has not been tested.

## Learning Algorithm

### Deep Q-Learning 
Deep Q-Learning (or DQN) uses a neural network (NN) to represent the action-value tupel q(s,a) for a given state s as an input to the NN. The algoritm adapts the weights of NN to target the (unknown) optimal action-value function _q*(s,a)_ from which we can extract optimal behaviour by choosing the action _a_ that maximizes the value _q*(a,s)_ at a given state _s_. The weight adaptation is done by gradient descent over the expected quadratic loss between the target action value function and the _predicted_ values by the NN. 
Since the target is unknown, we aproximate the target by the so-called Temporal-Difference target composed of the reward and the maximized action-value q(s,a) for the following state. Hence, we have an iterative procedure in which we evaluate the Q-values for the sampled states and update the NN using this information. 
In order to stabilize the learning procedure, it is required to fix the maximum Q-value for the target for some update steps. This is called _fixed Q-values_ and it decouples the target from the parameter updates. Notice that this approach requires that we initialize 2 different NNs, one used to sample the actions (the 'actor' network), and one to evaluate the targets (the 'target' network). Hence, to ensure that the actor and the target networks synchronize, we have also to update the target network in the direction of the actor network.
An additional characteristic of the vanilla DQN algorithm is the use of a 'replay buffer' in which the sampled state, action, reward and next state (S,A,R,S') _experience tuples_ are saved. Then, when updating the weights, a mini-batch is  sampled from this buffer uniformly to evalue the gradient and make the update. This _experience replay_ breaks the correlation between the samples used in the update step and stabilizes learning.
#### Pseudo-code
```
 Initialize experience replay buffer with capacity C
 Initialize NN with random weights
 Initialize epsilon
 score = 0
 for i_episode = 1,...,N:
     Initialize state
     for t = 1,...,T:
         Get epsilon-greedy action given state
         Get next_state and reward given action
         Check if the episode is done
         Save experience tuple in replay buffer
         Every K steps:
            Update agent's weights
         State = next_state
         Score += reward
         if done: 
            break             
        Decrease epsilon: epsilon = max(min_epsilon, decay*epsilon)
```

The epsilon-greedy action selection is the standard uniform draw of an action with probability epsilon and otherwise the action that maximizes the Q-values. 
The step above that needs to be clarifed is the agent's update. While at each step the resulting experience tuple is saved to the replay buffer, the gradient descent step is performed only every K steps. The update step is chosen to be ADAM. 
In addition, every time the actor network is updated, we update the target network by a small quantity _tau_ in the direction of the actor network, i.e. with an exponential moving average update.

__along with the chosen hyperparameters__

### The neural network architecture
The neural network used as agent for this report is a simple 2-hidden layers neural network with relu activation. The hidden layers have 50 neurons each and are fully connected between them, with the input layer (37x50) and the output layer (50x4). The weight initialization of each layer is the default initialization of nn.Linear in PyTorch. The output layer is simply a linear readout layer giving the 4 Q-values, one for each available action.


## Results
![Plot of Rewards](https://github.com/hcruiz/Navigation/blob/master/Score_smoothed_vs_Episode.png)

The plot above shows that the agent learned the correct actions to receive an average reward of +14 over 100 episodes. The condition of 
the environment to be solved was set to a higher averaged (+14) than the required. As shown in the figure, this target was met after 500
episodes. In addition it was observed, that the agent can get a reward up to +15 in less than 2000 episodes.


## Future Work

The performance of the vanilla DQN agent with similar parameters used in the LunarLander gym environment was surprisingly efficient. However, the learning profile hints at room for improvement. Here, we describe future changes to the vanilla implementations that will be explored:

* Thorough hyper-parameter search. A systematic grid search could be performed to obtain better parameters than the currently used and to study the behaviour of the agent depending on eps_end, eps_decay, gamma, tau (from least to most interesting).
* A more systematic exploration of the architectures are also be promising to improve performance.
* Implementation of Priority Experience Replay and tuning the hyper-parameters of this step.
* Implementation of the Double Q-Learning procedure is straightforward and a promising modification.
* Implementation of the Duelling Network Architecture could help if the Q-value function has a state-dependent offset and their dependent on the actions ('advantage values') is additive. The improvement of this modification might, however, be more significant if pixel values are used as state observations. 
* Finally, other modifications included in the Rainbow algorithm could help. Each of these modifications (Multi-step bootstrap targets, Distributional DQN and Noisy DQN) should be explored progressively. 
