import numpy as np
import copy
import random

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed=1, mu=0.0, theta=0.15, sigma=0.25):
        """Initialize parameters and noise process."""
        random.seed(seed)
        np.random.seed(seed)

        self.size = size
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.state = copy.copy(self.mu)

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = np.ones(self.size) * self.mu

    def noise(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state
