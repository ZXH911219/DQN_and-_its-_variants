import torch
import numpy as np
import pytorch_lightning as pl
import logging
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

from gridworld_env import RealGridWorld
from hw3_2_double_dueling_dqn import train_double_dueling_dqn_player_mode
from hw3_3_lightning_dqn import LightningDQN
from visualize import create_animation, plot_policy_and_value

def generate_multi_player():
    print("\n--- Training Player Mode ---")
    env = RealGridWorld(mode='player')
    model, _, _ = train_double_dueling_dqn_player_mode(env, episodes=500, batch_size=32)
    
    print("Generating 3 Player Mode Animations...")
    for i in range(1, 4):
        # Env reset in player mode randomizes ONLY the player position
        create_animation(env, model, f"nav_player_{i}.gif")

def generate_multi_random():
    print("\n--- Training Random Mode ---")
    env = RealGridWorld(mode='random')
    model = LightningDQN(input_dim=64, output_dim=4, env=env)
    
    trainer = pl.Trainer(
        max_epochs=150, 
        limit_train_batches=50,
        gradient_clip_val=1.0, 
        enable_progress_bar=False, 
        enable_model_summary=False, 
        logger=False
    )
    trainer.fit(model)
    
    print("Generating 3 Random Mode Animations & Maps...")
    for i in range(1, 4):
        # Env reset in random mode randomizes ALL objects
        env.reset() 
        create_animation(env, model, f"nav_random_{i}.gif")
        plot_policy_and_value(env, model, f"policy_value_random_{i}.png")

if __name__ == "__main__":
    generate_multi_player()
    generate_multi_random()
