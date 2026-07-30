import json
from pathlib import Path

try:
    import torch
    def load_checkpoint_payload(filepath):
        return torch.load(filepath)
except ImportError:
    import pickle
    def load_checkpoint_payload(filepath):
        with open(filepath, "rb") as f:
            return pickle.load(f)

OUTPUT_DIR = Path("/app/task/environment/app/output")
if not OUTPUT_DIR.exists():
    OUTPUT_DIR = Path(__file__).resolve().parent.parent / "environment" / "app" / "output"

def test_output_artifacts_exist():
    """Verify that all three required output artifacts exist under /app/task/environment/app/output."""
    metrics_file = OUTPUT_DIR / "final_metrics.json"
    checkpoint_file = OUTPUT_DIR / "model_checkpoint.pt"
    log_file = OUTPUT_DIR / "training.log"

    assert metrics_file.exists(), f"Missing output file: {metrics_file}"
    assert checkpoint_file.exists(), f"Missing output file: {checkpoint_file}"
    assert log_file.exists(), f"Missing output file: {log_file}"

def test_metrics_schema():
    """Verify final_metrics.json contains valid JSON matching the expected artifact schema."""
    metrics_file = OUTPUT_DIR / "final_metrics.json"
    with open(metrics_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "artifact_version" in data, "final_metrics.json missing 'artifact_version'"
    assert data["artifact_version"] == 1, f"Expected artifact_version 1, got {data['artifact_version']}"
    assert "metrics" in data, "final_metrics.json missing top-level 'metrics' key"

    m = data["metrics"]
    assert isinstance(m, dict), "'metrics' must be a dictionary"
    assert "episodes_completed" in m, "metrics missing 'episodes_completed'"
    assert "average_reward" in m, "metrics missing 'average_reward'"
    assert "buffer_size" in m, "metrics missing 'buffer_size'"
    assert m["episodes_completed"] == 100, f"Expected 100 episodes, got {m['episodes_completed']}"

def test_checkpoint_schema():
    """Verify model_checkpoint.pt matches the PyTorch state_dict checkpoint schema."""
    checkpoint_file = OUTPUT_DIR / "model_checkpoint.pt"
    data = load_checkpoint_payload(checkpoint_file)

    assert "artifact_version" in data, "model_checkpoint.pt missing 'artifact_version'"
    assert data["artifact_version"] == 1, f"Expected artifact_version 1, got {data['artifact_version']}"
    assert "state_dict" in data, "model_checkpoint.pt missing 'state_dict'"

    sd = data["state_dict"]
    assert isinstance(sd, dict), "'state_dict' must be a dictionary"
    assert "episodes_completed" in sd, "state_dict missing 'episodes_completed'"
    assert "buffer_size" in sd, "state_dict missing 'buffer_size'"
    assert "q_table" in sd, "state_dict missing 'q_table'"
    assert sd["episodes_completed"] == 100, f"Expected 100 episodes, got {sd['episodes_completed']}"

def test_training_log_format():
    """Verify training.log contains non-empty progress log messages."""
    log_file = OUTPUT_DIR / "training.log"
    content = log_file.read_text(encoding="utf-8")

    assert len(content.strip()) > 0, "training.log is empty"
    assert "Starting training run" in content, "training.log missing start message"
    assert "Completed 100 episodes" in content, "training.log missing completion message"
