import os
import sys
import subprocess
from pathlib import Path

APP_DIR = Path("/app/task/environment/app")
if not APP_DIR.exists():
    APP_DIR = Path(__file__).resolve().parent.parent / "environment" / "app"

# 1. Patch artifact_manager.py if needed
am_file = APP_DIR / "artifact_manager.py"
if am_file.exists():
    content = am_file.read_text(encoding="utf-8")
    if 'artifacts["output_dir"]' in content and 'artifacts.get(' not in content:
        content = content.replace('Path(artifacts["output_dir"])', 'Path(artifacts.get("output_dir", artifacts.get("artifact_dir", "output")))')
        am_file.write_text(content, encoding="utf-8")

# 2. Patch replay_buffer.py (global random -> self.rng)
rb_file = APP_DIR / "replay_buffer.py"
if rb_file.exists():
    content = rb_file.read_text(encoding="utf-8")
    content = content.replace("return random.sample(self.buffer, batch_size)", "return self.rng.sample(self.buffer, batch_size)")
    rb_file.write_text(content, encoding="utf-8")

# 3. Patch trainer.py (un-nest training_metrics & checkpoint payload structure)
tr_file = APP_DIR / "trainer.py"
if tr_file.exists():
    content = tr_file.read_text(encoding="utf-8")
    old_metrics = '''metrics = {
            "training_metrics": {
                "episodes_completed": episodes_completed,
                "average_reward": average_reward,
                "buffer_size": len(self.replay_buffer)
            }
        }'''
    new_metrics = '''metrics = {
            "episodes_completed": episodes_completed,
            "average_reward": average_reward,
            "buffer_size": len(self.replay_buffer)
        }'''
    content = content.replace(old_metrics, new_metrics)

    old_ckpt = '''checkpoint = {
            "checkpoint": {
                "episodes_completed": episodes_completed,
                "buffer_size": len(self.replay_buffer),
                "q_table": self.agent.q_table
            }
        }'''
    new_ckpt = '''checkpoint = {
            "episodes_completed": episodes_completed,
            "buffer_size": len(self.replay_buffer),
            "q_table": self.agent.q_table
        }'''
    content = content.replace(old_ckpt, new_ckpt)
    tr_file.write_text(content, encoding="utf-8")

# Execute training script with cwd set to APP_DIR
train_script = APP_DIR / "train.py"
subprocess.run([sys.executable, str(train_script)], cwd=str(APP_DIR), check=True)
