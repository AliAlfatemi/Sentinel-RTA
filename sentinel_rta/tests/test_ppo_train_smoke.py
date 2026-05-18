import pytest
import os
import sys
import subprocess

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.mark.expensive
def test_ppo_train_smoke():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_script = os.path.join(base_dir, "..", "scripts", "train_ppo.py")
    config_path = os.path.join(base_dir, "..", "configs", "ppo.yaml")
    output_dir = os.path.join(base_dir, "..", "results", "smoke_test_train")
    
    cmd = [
        sys.executable, train_script,
        "--config", config_path,
        "--seed", "42",
        "--output_dir", output_dir,
        "--total_timesteps", "64"
    ]
    
    subprocess.run(cmd, check=True)
    assert os.path.exists(os.path.join(output_dir, "model.zip"))
