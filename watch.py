import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt


from unityagents import UnityEnvironment
from drl.agent import Agent


# get environment
env = UnityEnvironment(file_name='Reacher_Linux/Reacher.x86_64')

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=False)[brain_name]

# number of agents
num_agents = len(env_info.agents)

# size of each action
action_size = brain.vector_action_space_size

# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]

agent = Agent(state_size=state_size, action_size=action_size, random_seed=1337)
agent.load(torch.load('checkpoints/checkpoint_actor_v2_solved.pth'), torch.load('checkpoints/checkpoint_critic_v2.pth'))

print_every=100
scores_deque = deque(maxlen=print_every)
scores = []
for i_episode in range(1, 100 + 1):
    env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
    state = env_info.vector_observations[0]  # get the current state (for each agent)
    score = 0
    for t in range(1000):
        action = agent.act(state)
        # get new state & reward & done_information
        env_info = env.step(action)[brain_name]
        next_state = env_info.vector_observations[0]  # get next state (for each agent)
        reward = env_info.rewards[0]  # get reward (for each agent)
        done = env_info.local_done[0]  # see if episode finished

        state = next_state
        score += reward
        if done:
            break

    scores_deque.append(score)
    scores.append(score)

    print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)), end="")
    if i_episode % print_every == 0:
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
