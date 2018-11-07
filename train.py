import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
#%matplotlib inline


from unityagents import UnityEnvironment
from drl.agent import Agent

# get environment
env = UnityEnvironment(file_name='Reacher_Linux/Reacher.x86_64')

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]

print(states)

# trains only one agent
#def train_one():
# init
env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
agent = Agent(state_size=state_size, action_size=action_size, random_seed=1337)


def ddpg(n_episodes=500, max_t=1000, print_every=100):
    scores_deque = deque(maxlen=print_every)
    scores = []
    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
        state = env_info.vector_observations[0]  # get the current state (for each agent)
        score = 0
        for t in range(max_t):
            action = agent.act(state)
            # get new state & reward & done_information
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]  # get next state (for each agent)
            reward = env_info.rewards[0]  # get reward (for each agent)
            done = env_info.local_done[0]  # see if episode finished

            agent.step(state, action, reward, next_state, done)

            state = next_state
            score += reward
            if done:
                break

        scores_deque.append(score)
        scores.append(score)

        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)), end="")
        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))

    torch.save(agent.actor_local.state_dict(), 'checkpoints/checkpoint_actor_v2_solved.pth')
    torch.save(agent.critic_local.state_dict(), 'checkpoints/checkpoint_critic_v2_solved.pth')
    return scores


scores = ddpg()

env.close()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores) + 1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
fig.savefig('checkpoints/scores_v2_solved.png')
plt.show()
