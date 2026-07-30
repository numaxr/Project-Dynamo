The reinforcement learning training pipeline under `/app/task/environment/app` regressed after an infrastructure migration and fails to produce the expected output artifacts. Diagnose and repair the pipeline code under `/app/task/environment/app`.

Do not modify files outside `/app/task/environment/app`.

When `/app/task/environment/app/train.py` is executed, it must generate the following three output artifacts in `/app/task/environment/app/output`:

1. `/app/task/environment/app/output/final_metrics.json`:
   A JSON file containing top-level keys `"artifact_version"` (integer `1`) and `"metrics"` (object).
   The `"metrics"` object must contain:
   - `"episodes_completed"`: Total episodes completed (integer `100`).
   - `"average_reward"`: Average episode reward (float).
   - `"buffer_size"`: Replay buffer size at completion (integer `500`).

2. `/app/task/environment/app/output/model_checkpoint.pt`:
   A PyTorch checkpoint file loaded via `torch.load()`, containing top-level keys `"artifact_version"` (integer `1`) and `"state_dict"` (object).
   The `"state_dict"` object must contain:
   - `"episodes_completed"`: Total episodes completed (integer `100`).
   - `"buffer_size"`: Replay buffer size at completion (integer `500`).
   - `"q_table"`: Dictionary of tabular Q-values mapping state indices to action value lists.

3. `/app/task/environment/app/output/training.log`:
   A plain text log file containing log statements for training progress, including:
   - `"Starting training run..."`
   - `"Completed 100 episodes..."`

You have 120 seconds to complete this task. Do not cheat by using online solutions or hints specific to this task.
