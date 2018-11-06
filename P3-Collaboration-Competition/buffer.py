from collections import deque, namedtuple
import numpy as np
import random
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, seed=1, 
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        random.seed(seed)
        np.random.seed(seed)
        self.device = device
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.field_names = ["state", "full_state", "action", "reward", "next_state", "full_next_state", "done"]
        self.experience = namedtuple("Experience", field_names=self.field_names)
            
    
    def add(self, state, full_state, action, reward, next_state, full_next_state, done):
        """Add a new experience to memory."""        
        e = self.experience(state, full_state, action, reward, next_state, full_next_state, done)
        self.memory.append(e)
    

    def sample(self, batch_size, tensor=True):
        """Randomly sample a batch of experiences from memory."""
        # samples = np.vstack(random.sample(self.memory, self.batch_size))

        experiences = random.sample(self.memory, batch_size)
        states = np.array([e.state for e in experiences if e is not None])
        full_states = np.array([e.full_state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None])
        rewards = np.array([e.reward for e in experiences if e is not None])
        next_states = np.array([e.next_state for e in experiences if e is not None])
        full_next_states = np.array([e.full_next_state for e in experiences if e is not None])
        dones = np.array([e.done for e in experiences if e is not None])

        if tensor:
            states = torch.from_numpy(states).float().to(self.device)
            full_states = torch.from_numpy(full_states).float().to(self.device)
            actions = torch.from_numpy(actions).float().to(self.device)
            rewards = torch.from_numpy(rewards).float().to(self.device)
            next_states = torch.from_numpy(next_states).float().to(self.device)
            full_next_states = torch.from_numpy(full_next_states).float().to(self.device)
            dones = torch.from_numpy(dones.astype(np.uint8)).float().to(self.device)
        
        return states, full_states, actions, rewards, next_states, full_next_states, dones


    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


    def size(self):
        return(len(self.memory))
