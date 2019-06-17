# Navigation in Bananas Collector Environment

This report provides a description of the Deep Q-Learning implementation to solve the navigation and recollection problem of the (Unity) Bananas Environment. 
The implementation is such that, in principle, it is compatible with any Unity environment, although this has not been tested.

## Learning Algorithm

The report clearly describes 
- the learning algorithm, along with the chosen hyperparameters
- the model architectures for any neural networks.

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
* Implementation of the Double Q-Learning precedure is straightforward and a promising modification.
* Implementation of the Duelling Network Architecture could help if the Q-value function has a state-dependent offset and their dependent on the actions ('advantage values') is additive. The improvement of this modification might, however, be more significant if pixel values are used as state observations. 
* Finally, other modifications included in the Rainbow algorithm could help. Each of these modifications (Multi-step bootstrap targets, Distributional DQN and Noisy DQN) should be explored progressively. 
