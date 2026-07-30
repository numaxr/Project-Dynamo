import random

class LineWorldAgent:
    """Simple tabular agent for LineWorld environment."""
    def __init__(self, n_states: int = 5, n_actions: int = 2, seed: int = 42):
        self.n_states = n_states
        self.n_actions = n_actions
        self.rng = random.Random(seed)
        self.q_table = {s: [0.0] * n_actions for s in range(n_states)}

    def select_action(self, state: int, epsilon: float = 0.1) -> int:
        if self.rng.random() < epsilon:
            return self.rng.randint(0, self.n_actions - 1)
        values = self.q_table.get(state, [0.0] * self.n_actions)
        max_v = max(values)
        best_actions = [a for a, v in enumerate(values) if v == max_v]
        return self.rng.choice(best_actions)

    def learn(self, state, action, reward, next_state, done, gamma=0.9, alpha=0.1):
        current_q = self.q_table[state][action]
        next_max = max(self.q_table[next_state]) if not done else 0.0
        target = reward + gamma * next_max
        self.q_table[state][action] += alpha * (target - current_q)
