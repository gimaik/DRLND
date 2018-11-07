from buffer import ReplayBuffer
from model import Actor, Critic
from OUNoise import OUNoise

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MultiAgent:

    def __init__(self, state_size, action_size, num_agents=2, eps_before_train=500, gamma=0.99, 
                 batch_size=128, buffer_size=int(1e5), lr_actor=1e-4, lr_critic=1e-3, weight_decay=0,
                 tau=1e-3, noise_weight=1.0, noise_decay=0.999998, noise_min=1e-3, seed=0, device="cuda:0"):


        # (self, state_size, action_size, num_agents=2, random_seed=1, lr_actor=2e-4, lr_critic=1e-3,
        #          weight_decay=0, tau=2e-3, device=device)

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents     
        self.action_dim = action_size * num_agents
        
        self.eps_before_train = eps_before_train
        self.gamma = gamma
        self.batch_size = batch_size
        self.buffer_size = buffer_size

        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.weight_decay = weight_decay
        self.tau = tau        
        
        self.noise_weight = noise_weight
        self.noise_decay = noise_decay
        self.noise_min = noise_min
        self.device = device
        self.i_episode = 0

        self.agents = [DDPG(self.state_size, self.action_size, self.num_agents, random_seed=2*i*seed, 
                        lr_actor=self.lr_actor, lr_critic=self.lr_critic, weight_decay=self.weight_decay, 
                        tau=self.tau, device=self.device) for i in range(self.num_agents)]
        self.memory = ReplayBuffer(self.action_size, self.buffer_size, seed)


    def reset(self):
        for agent in self.agents:
            agent.reset()


    def act(self, states, add_noise=True):
        noise_weight = self.noise_weight if add_noise else 0.0        

        if (self.i_episode >= self.eps_before_train) and (self.noise_weight > self.noise_min):
            self.noise_weight *= self.noise_decay
            noise_weight = self.noise_weight

        actions = [agent.act(s, noise_weight=noise_weight) for s, agent in zip(states, self.agents)]
        return np.array(actions)


    def step(self, states, actions, rewards, next_states, dones, t, i_episode):

        full_state = states.reshape(-1)
        full_next_state = next_states.reshape(-1)
        self.i_episode = i_episode

        self.memory.add(state=states, full_state=full_state, action=actions, reward=rewards,
                        next_state=next_states, full_next_state=full_next_state, done=dones)        

        if (i_episode >= self.eps_before_train) and (self.memory.size() >= self.batch_size):
            for agent_id in range(self.num_agents):
                experiences = self.memory.sample(self.batch_size)
                self.learn(agent_id, experiences)
            self.soft_update_all()


    def soft_update_all(self):
        for agent in self.agents:
            agent.soft_update_all()


    def learn(self, agent_id, experiences):
        agent = self.agents[agent_id]

        states_e, full_states_e, actions_e, rewards_e, next_states_e, full_next_states_e, dones_e = experiences
        rewards = rewards_e[:, agent_id].view(-1, 1)
        dones = dones_e[:, agent_id].view(-1, 1)

        # Update critic
        target_actions = self.target_act(next_states_e)
        Q_target_next = agent.critic_target(full_next_states_e, target_actions.view(-1, self.action_dim))
        Q_target = rewards + self.gamma * Q_target_next * (1.0 - dones)
        Q_local = agent.critic_local(full_states_e, actions_e.view(-1, self.action_dim))


        critic_loss = F.mse_loss(input=Q_local, target=Q_target.detach())
        agent.critic_local.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.critic_local.parameters(), 2)
        agent.critic_optimizer.step()

        # Update the actor policy
        agent_states = states_e[:, agent_id]
        agent_actions = agent.actor_local(agent_states)
        actions = actions_e.clone()
        actions[:, agent_id] = agent_actions

        actor_loss = -agent.critic_local(full_states_e, actions.view(-1, self.action_dim)).mean()
        agent.actor_local.zero_grad()
        torch.nn.utils.clip_grad_norm_(agent.actor_local.parameters(), 2)
        actor_loss.backward()
        agent.actor_optimizer.step()

        actor_loss_value = actor_loss.cpu().detach().item()
        critic_loss_value = critic_loss.cpu().detach().item()
        return actor_loss_value, critic_loss_value


    def target_act(self, states):
        actions = torch.zeros(states.shape[:2] + (self.action_size,), dtype=torch.float, device=self.device)
        for i in range(self.num_agents):
            actions[:, i, :] = self.agents[i].actor_target(states[:, i])
        return actions


    def local_act(self, states):
        actions = torch.zeros(states.shape[:2] + (self.action_size,), dtype=torch.float, device=self.device)
        for i in range(self.num_agents):
            actions[:, i, :] = self.agents[i].actor_local(states[:, i])
        return actions



class DDPG():

    def __init__(self, state_size, action_size, num_agents=2, random_seed=1, lr_actor=2e-4, lr_critic=1e-3,
                 weight_decay=0, tau=2e-3, device=device):
        self.num_agents = num_agents
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random_seed
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.weight_decay = weight_decay
        self.tau = tau
        self.device = device
        self.noise = OUNoise(self.action_size, self.seed)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(self.state_size, self.action_size, self.seed).to(device)
        self.actor_target = Actor(self.state_size, self.action_size, self.seed).to(device)
        
        # Critic Network (w/ Target Network)
        self.critic_local = Critic(self.state_size*self.num_agents, self.action_size*self.num_agents, self.seed).to(device)
        self.critic_target = Critic(self.state_size*self.num_agents, self.action_size*self.num_agents, self.seed).to(device)
        
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.lr_actor)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.lr_critic, weight_decay=self.weight_decay)

        # Setting same initialization for both the targets and original networks
        self.hard_update_all()


    def act(self, states, noise_weight=1.0):
        states = torch.from_numpy(states).float().to(device=self.device).view(-1, self.state_size)
        self.actor_local.eval()
        with torch.no_grad():
            actions = self.actor_local(states).cpu().data.numpy()
        self.actor_local.train()

        actions += noise_weight*self.noise.noise()
        return np.clip(np.squeeze(actions), -1, 1)


    def act_target(self, states, noise_weight=1.0):
        states = torch.from_numpy(states).float().to(device=self.device).view(-1, self.state_size)
        self.actor_target.eval()
        with torch.no_grad():
            actions = self.actor_target(states).cpu().data.numpy()
        self.actor_local.train()
        actions += noise_weight*self.noise.noise()
        return np.clip(actions, -1, 1)

    def reset(self):
        self.noise.reset()

    def soft_update_all(self):
        self.soft_update(local_model = self.actor_local, target_model = self.actor_target, tau = self.tau)
        self.soft_update(local_model = self.critic_local, target_model = self.critic_target, tau = self.tau)

    def hard_update_all(self):
        self.hard_update(source = self.actor_local, target = self.actor_target)
        self.hard_update(source = self.critic_local, target = self.critic_target)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)