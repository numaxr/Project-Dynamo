import random
from collections import deque

class ReplayBuffer:
    """Experience replay buffer for RL agent."""
    def __init__(self, capacity: int = 500, seed: int = 42):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        return self.rng.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
