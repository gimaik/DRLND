
from collections import deque
from itertools import count
import time
import torch
import numpy as np

def train_agent(agent, env, brain_name,max_score, n_episodes=30000, max_t = 2000, print_every=500):
    
    """Deep Deterministic Policy Gradient.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores_deque = deque(maxlen=100) # list containing scores from each episode
    scores_global = []

    for i_episode in range(1, n_episodes+1):
        
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        scores = np.zeros(agent.num_agents)
        agent.reset()
        scores_average = 0
        timestep = time.time()
        
        for t in range(max_t):
            actions = agent.act(states, add_noise=True)            
            env_info = env.step(actions)[brain_name]             
            next_states = env_info.vector_observations            
            rewards = np.array(env_info.rewards)
            dones = np.array(env_info.local_done)
            agent.step(states, actions, rewards, next_states, dones, t, i_episode)
            states = next_states
            scores += rewards
            
            if np.any(dones):
                break                 
                
        score = np.max(scores)
        scores_deque.append(score)
        scores_global.append(score)        
        scores_average = np.mean(scores_deque)
        if i_episode  % 100 ==0:
            print('\rEpisode {}\t Score: {:.3f}, Mean: {:.2f}, Max: {:.2f}, Min: {:.2f}, Time: {:.2f}'
                  .format(i_episode, score, np.mean(scores_deque), np.max(scores_deque), np.min(scores_deque),time.time() - timestep))
        
        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}\n'.format(i_episode, np.mean(scores_deque)))
            
            for i, a in enumerate(agent.agents):
                torch.save(a.critic_local.state_dict(), 'checkpoint_critic_' + str(i) + '_' + str(i_episode) + '.pth')
                torch.save(a.actor_local.state_dict(), 'checkpoint_actor_' + str(i) + '_' + str(i_episode) + '.pth')
        
        if scores_average >= max_score:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_deque)))
            for i, a in enumerate(agent.agents):
                torch.save(a.critic_local.state_dict(), 'checkpoint_critic_' + str(i) + '.pth')
                torch.save(a.actor_local.state_dict(), 'checkpoint_actor_' + str(i) + '.pth')
            break
            
    return scores_global

def load_weights(agent, actor_weights, critic_weights):
    for agent_id, a in enumerate(agent.agents):
        a.actor_local.load_state_dict(torch.load(actor_weights[agent_id]))
        a.critic_local.load_state_dict(torch.load(critic_weights[agent_id]))        
    return agent