import json
from pathlib import Path

try:
    import torch
    def save_checkpoint_payload(payload, filepath):
        torch.save(payload, filepath)
except ImportError:
    import pickle
    def save_checkpoint_payload(payload, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(payload, f)

class ArtifactManager:
    """Manages saving training metrics, model checkpoints, and logs."""
    def __init__(self, config: dict):
        artifacts = config.get("artifacts", {})
        output_dir = artifacts.get("output_dir", artifacts.get("artifact_dir", "output"))
        self.output_dir = Path(output_dir)
        self.metrics_file = self.output_dir / artifacts.get("metrics_file", "final_metrics.json")
        self.checkpoint_file = self.output_dir / artifacts.get("checkpoint_file", "model_checkpoint.pt")
        self.log_file = self.output_dir / artifacts.get("log_file", "training.log")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_metrics(self, metrics: dict):
        payload = {
            "artifact_version": 1,
            "metrics": metrics
        }
        with open(self.metrics_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def save_checkpoint(self, checkpoint: dict):
        payload = {
            "artifact_version": 1,
            "state_dict": checkpoint
        }
        save_checkpoint_payload(payload, self.checkpoint_file)

    def log(self, message: str):
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(message + "\n")
