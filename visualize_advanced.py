import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
import pytorch_lightning as pl
import logging
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

from gridworld_env import RealGridWorld
from hw3_2_double_dueling_dqn import train_double_dueling_dqn_player_mode
from hw3_3_lightning_dqn import LightningDQN
from visualize import create_animation, plot_policy_and_value

def run_player_mode():
    print("\n--- Training Player Mode (Double/Dueling DQN) ---")
    env = RealGridWorld(mode='player')
    # Train for more episodes so it learns all starting positions
    model, _ = train_double_dueling_dqn_player_mode(env, episodes=600, batch_size=32)
    
    print("Generating Player Mode Visuals...")
    create_animation(env, model, "navigation_player.gif")
    plot_policy_and_value(env, model, "policy_value_player.png")

def run_random_mode():
    print("\n--- Training Random Mode (PyTorch Lightning) ---")
    env = RealGridWorld(mode='random')
    model = LightningDQN(input_dim=64, output_dim=4, env=env)
    
    # Random mode takes a LOT of data to generalize. 
    # limit_train_batches is required because our RLDataset is infinite.
    trainer = pl.Trainer(
        max_epochs=150, 
        limit_train_batches=50,
        gradient_clip_val=1.0, 
        enable_progress_bar=False, 
        enable_model_summary=False, 
        logger=False
    )
    trainer.fit(model)
    
    print("Generating Random Mode Visuals...")
    # Lock in one specific random layout for the visualization
    state = env.reset() 
    
    # plot_policy_and_value expects a normal PyTorch model, but LightningModule has forward() which works identically.
    # It will use the fixed layout currently in the environment.
    create_animation(env, model, "navigation_random.gif")
    plot_policy_and_value(env, model, "policy_value_random.png")

if __name__ == "__main__":
    # run_player_mode() # Already completed
    run_random_mode()
