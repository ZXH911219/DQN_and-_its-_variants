import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

# ==========================================
# 1. Experience Replay Buffer
# ==========================================
class ReplayBuffer:
    def __init__(self, capacity=1000):
        """
        Initialize the Replay Buffer.
        :param capacity: Maximum number of experiences to hold (1000 as requested)
        """
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        Add a new experience to the buffer.
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Sample a random batch of experiences from the buffer.
        """
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

# ==========================================
# 2. Naive DQN Architecture
# ==========================================
class NaiveDQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        """
        A basic Multilayer Perceptron (MLP) for DQN.
        """
        super(NaiveDQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        """
        Forward pass to compute Q-values for all actions.
        """
        return self.net(x)

# ==========================================
# 3. Training Logic (Static Mode)
# ==========================================
def train_naive_dqn_static_mode(env, episodes=500, batch_size=32, gamma=0.99, lr=1e-3):
    """
    Training loop for Naive DQN in Static Mode GridWorld.
    Note: 'env' should be a gym-like environment.
    """
    # Assuming the environment has these attributes
    input_dim = env.observation_space.shape[0] if hasattr(env, 'observation_space') else 64  # Example 8x8 grid
    output_dim = env.action_space.n if hasattr(env, 'action_space') else 4                   # 4 actions (up, down, left, right)
    
    model = NaiveDQN(input_dim, output_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    replay_buffer = ReplayBuffer(capacity=1000)
    
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995

    rewards_history = []
    loss_history = []

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        episode_losses = []

        while not done:
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = random.randint(0, output_dim - 1)
            else:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = model(state_tensor)
                action = q_values.argmax().item()

            # Step in environment
            next_state, reward, done, _ = env.step(action)
            
            # Store transition
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            # Train if buffer has enough samples
            if len(replay_buffer) >= batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                
                states = torch.FloatTensor(states)
                actions = torch.LongTensor(actions)
                rewards = torch.FloatTensor(rewards)
                next_states = torch.FloatTensor(next_states)
                dones = torch.FloatTensor(dones)

                # Compute current Q values
                q_values = model(states)
                current_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

                # Compute Target Q values
                with torch.no_grad():
                    next_q_values = model(next_states)
                    max_next_q = next_q_values.max(1)[0]
                    target_q = rewards + gamma * max_next_q * (1 - dones)

                # Compute loss and update weights
                loss = loss_fn(current_q, target_q)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                episode_losses.append(loss.item())

        # Decay epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        rewards_history.append(total_reward)
        avg_loss = sum(episode_losses)/len(episode_losses) if episode_losses else 0.0
        loss_history.append(avg_loss)
        
        if episode % 50 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}, Loss: {avg_loss:.4f}")

    return model, rewards_history, loss_history

if __name__ == "__main__":
    from dummy_env import DummyGridWorld
    env = DummyGridWorld(mode='static')
    print("Starting HW3-1 Training (50 episodes)...")
    model, rewards, losses = train_naive_dqn_static_mode(env, episodes=50, batch_size=16)
    print("HW3-1 Training Complete. Final Reward:", rewards[-1])
