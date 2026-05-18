import sys
import os
import subprocess

def test_phase2f_stress_smoke():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    stress_script = os.path.join(base_dir, "..", "scripts", "run_temporal_shield_stress.py")
    config_path = os.path.join(base_dir, "..", "configs", "ppo_c1_temporal.yaml")
    output_dir = os.path.join(base_dir, "..", "results", "phase2f_temporal_stress_smoke")
    
    cmd = [
        sys.executable, stress_script,
        "--config", config_path,
        "--seeds", "1",
        "--scenarios", "long_sustained_attack",
        "--policies", "edge_riding", "aggressive",
        "--episode_length", "50",
        "--output_dir", output_dir
    ]
    
    subprocess.run(cmd, check=True)
    
    # Verify outputs
    assert os.path.exists(os.path.join(output_dir, "temporal_stress_summary.csv"))
    assert os.path.exists(os.path.join(output_dir, "per_step_logs.csv"))
