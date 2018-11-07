# Implementation

For our agent to solve the environment we implemented the DDPG architecture to also predict continous actions, as it is needed for our two jointed arm.
Since Deep Reinforcement Learning(DRL) agents are likely to overestimate, or at least in my case, I implemented the L2 regularization with weight decay to prevent him from doing so.
Additionally target networks have been implemented for both the actor and the critic.
Both are updated with a soft-update. 
Furthermore the agent is only trained all 20 timesteps with 10 experiences from the replay buffer at once, to have more diversity between the updates.

# Hyperparameters

BUFFER_SIZE     = int(1e5)  # replay buffer size
BATCH_SIZE      = 128       # minibatch size
GAMMA           = 0.99      # discount factor
TAU             = 1e-3      # for soft update of target parameters
LR_ACTOR        = 1e-4      # learning rate of the actor
LR_CRITIC       = 1e-3      # learning rate of the critic
WEIGHT_DECAY    = 0.00001   # L2 weight decay

n_episodes  = 500         # maximum episodes to take 
max_t       = 1000              # maximum steps to take in one episodes


# Archtitecture

The architecture used is a DDPG. This architecture consists of an actor and an critic. 
This architecture is used because it can predict actions in an continuous action space.
Both models are set up with 2 hidden layers. The first hidden layer has the size 400 and the second the size 300.

# Score

The score needs to be over +30 over 100 consecutive episodes to solve the environment.
You can see the solved reward plot in the "checkpoints" folder with the naming "-solved".


# Future

The agent so far is pretty good in a very short learning cycle.
Although additional improvements can be made, like the Generalized Advantage Estimation(GAE) to improve the prediction of the expected return.
