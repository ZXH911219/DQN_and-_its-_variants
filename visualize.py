import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
from gridworld_env import RealGridWorld
from hw3_1_naive_dqn import train_naive_dqn_static_mode

def render_frame(env):
    """Renders the current state of the GridWorld to a numpy RGB array."""
    grid = np.ones((4, 4, 3)) # White background
    
    # Wall -> Black
    grid[env.wall_pos[0], env.wall_pos[1]] = [0.2, 0.2, 0.2]
    # Pit -> Red
    grid[env.pit_pos[0], env.pit_pos[1]] = [0.8, 0.1, 0.1]
    # Goal -> Green
    grid[env.goal_pos[0], env.goal_pos[1]] = [0.1, 0.8, 0.1]
    # Player -> Blue
    grid[env.player_pos[0], env.player_pos[1]] = [0.1, 0.1, 0.8]
    
    return grid

def create_animation(env, model, filename="navigation.gif"):
    state = env.reset()
    frames = []
    
    fig, ax = plt.subplots(figsize=(4, 4))
    plt.axis('off')
    
    done = False
    while not done:
        frames.append([ax.imshow(render_frame(env), animated=True)])
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = model(state_tensor)
        action = q_values.argmax().item()
        
        state, reward, done, _ = env.step(action)
    
    # Append final frame
    frames.append([ax.imshow(render_frame(env), animated=True)])
    
    ani = animation.ArtistAnimation(fig, frames, interval=400, blit=True, repeat_delay=1000)
    ani.save(filename, writer='pillow')
    plt.close()
    print(f"Saved animation to {filename}")

def plot_policy_and_value(env, model, filename="policy_value_maps.png"):
    value_map = np.zeros((4, 4))
    policy_map = np.zeros((4, 4), dtype=int)
    
    for r in range(4):
        for c in range(4):
            if (r, c) == env.wall_pos:
                value_map[r, c] = 0
                policy_map[r, c] = -1
                continue
                
            # Temporarily move player to this state to evaluate it
            env.player_pos = (r, c)
            state = env._get_state()
            
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = model(state_tensor)
                
            value_map[r, c] = q_values.max().item()
            policy_map[r, c] = q_values.argmax().item()

    # Draw the plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. Value Map Heatmap
    sns.heatmap(value_map, annot=True, cmap="YlGnBu", ax=axes[0], cbar=True, fmt=".2f")
    axes[0].set_title("Value Function V(s)")
    
    # 2. Policy Map with Arrows
    sns.heatmap(value_map, cmap="YlGnBu", ax=axes[1], cbar=False)
    axes[1].set_title("Optimal Policy $\pi(s)$")
    
    action_to_arrow = {0: '↑', 1: '↓', 2: '←', 3: '→'}
    
    for r in range(4):
        for c in range(4):
            if (r, c) == env.wall_pos:
                axes[1].text(c + 0.5, r + 0.5, "WALL", ha='center', va='center', color='white', weight='bold')
            elif (r, c) == env.goal_pos:
                axes[1].text(c + 0.5, r + 0.5, "GOAL", ha='center', va='center', color='green', weight='bold')
            elif (r, c) == env.pit_pos:
                axes[1].text(c + 0.5, r + 0.5, "PIT", ha='center', va='center', color='red', weight='bold')
            else:
                arrow = action_to_arrow[policy_map[r, c]]
                axes[1].text(c + 0.5, r + 0.5, arrow, ha='center', va='center', fontsize=20)
                
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved policy and value maps to {filename}")

if __name__ == "__main__":
    print("Training Naive DQN on Real GridWorld...")
    env = RealGridWorld(mode='static')
    # Train for 300 episodes to ensure convergence on static map
    model, _ = train_naive_dqn_static_mode(env, episodes=300, batch_size=32)
    
    print("Generating Visualizations...")
    create_animation(env, model, "navigation.gif")
    plot_policy_and_value(env, model, "policy_value_maps.png")
