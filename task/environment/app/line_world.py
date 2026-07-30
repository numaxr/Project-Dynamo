class LineWorldEnv:
    """1D LineWorld environment."""
    def __init__(self, n_states: int = 5):
        self.n_states = n_states
        self.state = 0

    def reset(self):
        self.state = 0
        return self.state

    def step(self, action: int):
        # 0: left, 1: right
        if action == 0:
            self.state = max(0, self.state - 1)
        else:
            self.state = min(self.n_states - 1, self.state + 1)
        
        done = (self.state == self.n_states - 1)
        reward = 1.0 if done else 0.0
        return self.state, reward, done, {}
