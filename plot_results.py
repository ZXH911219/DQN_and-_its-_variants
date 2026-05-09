import matplotlib.pyplot as plt
from hw3_1_naive_dqn import train_naive_dqn_static_mode
from hw3_2_double_dueling_dqn import train_double_dueling_dqn_player_mode
from dummy_env import DummyGridWorld

def main():
    episodes = 100 # Keep it reasonable so it runs fast
    print("Training HW3-1 (Naive DQN) for plotting...")
    env1 = DummyGridWorld(mode='static')
    _, rewards1 = train_naive_dqn_static_mode(env1, episodes=episodes, batch_size=16)

    print("Training HW3-2 (Double & Dueling DQN) for plotting...")
    env2 = DummyGridWorld(mode='player')
    _, rewards2 = train_double_dueling_dqn_player_mode(env2, episodes=episodes, batch_size=16)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(rewards1, label='HW3-1: Naive DQN (Static Mode)', alpha=0.7)
    plt.plot(rewards2, label='HW3-2: Double/Dueling DQN (Player Mode)', alpha=0.7)
    
    # Calculate moving averages for smoother lines
    window = 10
    if len(rewards1) >= window:
        ma1 = [sum(rewards1[i:i+window])/window for i in range(len(rewards1)-window+1)]
        ma2 = [sum(rewards2[i:i+window])/window for i in range(len(rewards2)-window+1)]
        plt.plot(range(window-1, episodes), ma1, color='blue', linestyle='--', label='Naive DQN (Moving Avg)')
        plt.plot(range(window-1, episodes), ma2, color='orange', linestyle='--', label='Double/Dueling DQN (Moving Avg)')

    plt.title('Training Rewards over Episodes (Dummy Environment)')
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig('results_chart.png')
    print("Chart saved to results_chart.png")

if __name__ == "__main__":
    main()
