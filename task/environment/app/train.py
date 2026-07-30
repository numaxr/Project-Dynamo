import sys
from pathlib import Path
import yaml

app_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(app_dir))

from trainer import LineWorldTrainer

def main():
    config_path = app_dir / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    trainer = LineWorldTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()
