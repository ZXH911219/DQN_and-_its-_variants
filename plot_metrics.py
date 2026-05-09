import matplotlib.pyplot as plt
import pytorch_lightning as pl
import logging
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

from dummy_env import DummyGridWorld
from hw3_1_naive_dqn import train_naive_dqn_static_mode
from hw3_2_double_dueling_dqn import train_double_dueling_dqn_player_mode
from hw3_3_lightning_dqn import LightningDQN
from hw3_4_rainbow_dqn import train_rainbow_dqn_random_mode

def plot_and_save(rewards, losses, title, filename):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot Rewards
    axes[0].plot(rewards, color='blue', alpha=0.6, label='Raw Data')
    window = 10
    if len(rewards) >= window:
        ma = [sum(rewards[i:i+window])/window for i in range(len(rewards)-window+1)]
        axes[0].plot(range(window-1, len(rewards)), ma, color='darkblue', linestyle='--', label='10-Ep Moving Avg')
    axes[0].set_title(f"{title} - Total Reward per Episode")
    axes[0].set_ylabel("Total Reward")
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot Losses
    axes[1].plot(losses, color='red', alpha=0.6, label='Raw Data')
    if len(losses) >= window:
        ma_loss = [sum(losses[i:i+window])/window for i in range(len(losses)-window+1)]
        axes[1].plot(range(window-1, len(losses)), ma_loss, color='darkred', linestyle='--', label='10-Ep Moving Avg')
    axes[1].set_title(f"{title} - Average Loss per Episode")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved {filename}")

def main():
    episodes = 400
    
    # HW3-1
    print("Running HW3-1...")
    env1 = DummyGridWorld(mode='static')
    _, r1, l1 = train_naive_dqn_static_mode(env1, episodes=episodes, batch_size=16)
    plot_and_save(r1, l1, "HW3-1 Naive DQN", "hw3_1_metrics.png")
    
    # HW3-2
    print("Running HW3-2...")
    env2 = DummyGridWorld(mode='player')
    _, r2, l2 = train_double_dueling_dqn_player_mode(env2, episodes=episodes, batch_size=16)
    plot_and_save(r2, l2, "HW3-2 Double/Dueling DQN", "hw3_2_metrics.png")
    
    # HW3-3
    print("Running HW3-3...")
    env3 = DummyGridWorld(mode='random')
    model3 = LightningDQN(input_dim=64, output_dim=4, env=env3)
    trainer = pl.Trainer(max_epochs=episodes, limit_train_batches=10, gradient_clip_val=1.0, enable_progress_bar=False, enable_model_summary=False, logger=False)
    trainer.fit(model3)
    r3, l3 = model3.rewards_history, model3.loss_history
    plot_and_save(r3, l3, "HW3-3 PyTorch Lightning DQN", "hw3_3_metrics.png")
    
    # HW3-4
    print("Running HW3-4...")
    env4 = DummyGridWorld(mode='random')
    _, r4, l4 = train_rainbow_dqn_random_mode(env4, episodes=episodes, batch_size=16)
    plot_and_save(r4, l4, "HW3-4 Rainbow DQN", "hw3_4_metrics.png")

if __name__ == "__main__":
    main()
