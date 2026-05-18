import pytest
import os
import sys
import subprocess

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.mark.expensive
def test_reward_sweep_smoke():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sweep_script = os.path.join(base_dir, "..", "scripts", "run_reward_sweep.py")
    config_path = os.path.join(base_dir, "..", "configs", "reward_sweep.yaml")
    output_dir = os.path.join(base_dir, "..", "results", "smoke_test_reward_sweep")
    
    cmd = [
        sys.executable, sweep_script,
        "--config", config_path,
        "--output_dir", output_dir,
        "--seeds", "1",
        "--total_timesteps", "64",
        "--evaluation_episodes", "1",
        "--variants", "A" # Only run variant A for smoke test
    ]
    
    subprocess.run(cmd, check=True)
    assert os.path.exists(os.path.join(output_dir, "reward_sweep_summary.csv"))
