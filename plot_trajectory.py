import torch
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import logging
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

from gridworld_env import RealGridWorld
from hw3_2_double_dueling_dqn import train_double_dueling_dqn_player_mode
from hw3_3_lightning_dqn import LightningDQN
from hw3_4_rainbow_dqn import train_rainbow_dqn_random_mode

def plot_static_trajectory(env, model, filename):
    state = env._get_state()
    path = [env.player_pos]
    
    done = False
    max_steps = 20
    steps = 0
    while not done and steps < max_steps:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        if next(model.parameters()).is_cuda:
            state_tensor = state_tensor.cuda()
            
        with torch.no_grad():
            q_values = model(state_tensor)
        action = q_values.argmax().item()
        
        state, reward, done, _ = env.step(action)
        path.append(env.player_pos)
        steps += 1

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(3.5, -0.5)
    
    ax.set_xticks(np.arange(-0.5, 4, 1))
    ax.set_yticks(np.arange(-0.5, 4, 1))
    ax.grid(color='gray', linestyle='-', linewidth=2)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    for r in range(4):
        for c in range(4):
            if (r, c) == env.wall_pos:
                ax.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1, color='black'))
                ax.text(c, r, 'W', color='white', ha='center', va='center', fontsize=20, weight='bold')
            elif (r, c) == env.pit_pos:
                ax.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1, color='darkred'))
                ax.text(c, r, 'Pit', color='white', ha='center', va='center', fontsize=20, weight='bold')
            elif (r, c) == env.goal_pos:
                ax.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1, color='darkgreen'))
                ax.text(c, r, 'Goal', color='white', ha='center', va='center', fontsize=16, weight='bold')
                
    start_r, start_c = path[0]
    ax.text(start_c, start_r, 'S', color='blue', ha='center', va='center', fontsize=24, weight='bold')
    
    for i in range(len(path) - 1):
        r1, c1 = path[i]
        r2, c2 = path[i+1]
        if (r1, c1) != (r2, c2):
            ax.annotate("", xy=(c2, r2), xytext=(c1, r1),
                        arrowprops=dict(arrowstyle="->,head_length=0.8,head_width=0.4", color="blue", lw=3))
            
    plt.title("Agent Trajectory (Start $\\rightarrow$ Goal)")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved {filename}")

def main():
    print("Training Rainbow Mode for Trajectories...")
    env_rainbow = RealGridWorld(mode='random')
    model_rainbow, _, _ = train_rainbow_dqn_random_mode(env_rainbow, episodes=200, batch_size=32)
    
    for i in range(1, 4):
        success = False
        while not success:
            env_rainbow.reset()
            test_env = RealGridWorld(mode='random')
            test_env.goal_pos = env_rainbow.goal_pos
            test_env.pit_pos = env_rainbow.pit_pos
            test_env.wall_pos = env_rainbow.wall_pos
            test_env.player_pos = env_rainbow.player_pos
            
            done = False
            steps = 0
            while not done and steps < 20:
                tensor = torch.FloatTensor(test_env._get_state()).unsqueeze(0)
                with torch.no_grad():
                    q = model_rainbow(tensor)
                action = q.argmax().item()
                _, reward, done, _ = test_env.step(action)
                steps += 1
                if reward == 10.0:
                    success = True
                    break
        
        plot_static_trajectory(env_rainbow, model_rainbow, f"trajectory_rainbow_{i}.png")

if __name__ == "__main__":
    main()
