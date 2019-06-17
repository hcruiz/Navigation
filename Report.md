# Report on Navigation in Bananas Collector Environment

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

The submission has concrete future ideas for improving the agent's performance.
