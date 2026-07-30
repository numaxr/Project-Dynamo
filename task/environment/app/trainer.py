import time
from line_world import LineWorldEnv
from agent import LineWorldAgent
from replay_buffer import ReplayBuffer
from artifact_manager import ArtifactManager

class LineWorldTrainer:
    """Trainer for LineWorld agent."""
    def __init__(self, config: dict):
        self.config = config
        env_cfg = config.get("environment", {})
        train_cfg = config.get("training", {})

        self.n_states = env_cfg.get("n_states", 5)
        self.seed = env_cfg.get("seed", 42)
        self.episodes = train_cfg.get("episodes", 100)
        self.max_steps = train_cfg.get("max_steps", 20)
        self.epsilon = train_cfg.get("epsilon", 0.1)
        self.learning_rate = train_cfg.get("learning_rate", 0.1)
        self.gamma = train_cfg.get("gamma", 0.9)
        self.buffer_capacity = train_cfg.get("buffer_capacity", 500)

        self.env = LineWorldEnv(n_states=self.n_states)
        self.agent = LineWorldAgent(n_states=self.n_states, seed=self.seed)
        self.replay_buffer = ReplayBuffer(capacity=self.buffer_capacity, seed=self.seed)
        self.artifact_manager = ArtifactManager(config)

    def train(self):
        self.artifact_manager.log("Starting training run...")
        total_reward = 0.0
        episodes_completed = 0

        for episode in range(self.episodes):
            state = self.env.reset()
            episode_reward = 0.0

            for step in range(self.max_steps):
                action = self.agent.select_action(state, epsilon=self.epsilon)
                next_state, reward, done, _ = self.env.step(action)
                self.agent.learn(state, action, reward, next_state, done, gamma=self.gamma, alpha=self.learning_rate)
                self.replay_buffer.add(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward
                if done:
                    break

            total_reward += episode_reward
            episodes_completed += 1

        average_reward = total_reward / episodes_completed if episodes_completed > 0 else 0.0

        metrics = {
            "episodes_completed": episodes_completed,
            "average_reward": average_reward,
            "buffer_size": len(self.replay_buffer)
        }
        self.artifact_manager.save_metrics(metrics)

        checkpoint = {
            "episodes_completed": episodes_completed,
            "buffer_size": len(self.replay_buffer),
            "q_table": self.agent.q_table
        }
        self.artifact_manager.save_checkpoint(checkpoint)
        self.artifact_manager.log(f"Completed {episodes_completed} episodes with avg reward {average_reward:.2f}.")
