import pytest
import os
import sys
import subprocess

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.mark.expensive
def test_ppo_eval_smoke():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    eval_script = os.path.join(base_dir, "..", "scripts", "evaluate_ppo.py")
    config_path = os.path.join(base_dir, "..", "configs", "ppo.yaml")
    model_path = os.path.join(base_dir, "..", "results", "smoke_test_train", "model.zip")
    output_dir = os.path.join(base_dir, "..", "results", "smoke_test_eval")
    
    if not os.path.exists(model_path):
        return # Skip if train smoke failed or wasn't run
        
    cmd = [
        sys.executable, eval_script,
        "--config", config_path,
        "--model_path", model_path,
        "--seed", "42",
        "--shield", "off",
        "--output_dir", output_dir,
        "--eval_episodes", "1"
    ]
    
    subprocess.run(cmd, check=True)
    assert os.path.exists(os.path.join(output_dir, "episode_summary.csv"))
