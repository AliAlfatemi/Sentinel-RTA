import os
import sys
import yaml
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from envs.ddos_env import DDoSEnv

def train_ppo(config_path, seed, output_dir, total_timesteps=None):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        
    timesteps = total_timesteps if total_timesteps else config['training']['total_timesteps']
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save a copy of the config
    with open(os.path.join(output_dir, 'train_config.yaml'), 'w') as file:
        yaml.dump(config, file)
        
    # Create vectorized environment
    env = make_vec_env(lambda: DDoSEnv(max_steps=config['env']['max_steps'], 
                                       base_legitimate_rate=config['env']['base_legitimate_rate'],
                                       reward_mode=config['env'].get('reward_mode', 'A'),
                                       reward_weights=config['env'].get('reward_weights', {})), 
                       n_envs=1, seed=seed)
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=config['training']['learning_rate'],
        n_steps=config['training']['n_steps'],
        batch_size=config['training']['batch_size'],
        ent_coef=config['training']['ent_coef'],
        verbose=1,
        seed=seed
    )
    
    print(f"Training PPO for {timesteps} timesteps (Seed: {seed})...")
    model.learn(total_timesteps=timesteps)
    
    model_path = os.path.join(output_dir, "model.zip")
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--total_timesteps", type=int, default=None)
    args = parser.parse_args()
    
    train_ppo(args.config, args.seed, args.output_dir, args.total_timesteps)
