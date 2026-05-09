import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import copy

# ==========================================
# 1. Experience Replay Buffer (reused)
# ==========================================
class ReplayBuffer:
    def __init__(self, capacity=1000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

# ==========================================
# 2. Dueling DQN Architecture
# ==========================================
class DuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        """
        Dueling DQN separates the network into two streams:
        1. Value stream V(s): How good is it to be in this state?
        2. Advantage stream A(s, a): How good is this action compared to other actions?
        """
        super(DuelingDQN, self).__init__()
        
        # Shared feature extraction layer
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Stream 1: Value V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) # Outputs a single value per state
        )
        
        # Stream 2: Advantage A(s, a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim) # Outputs one advantage per action
        )

    def forward(self, x):
        features = self.feature_layer(x)
        
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Combine streams: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        # Subtracting the mean helps with identifiability of V and A.
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

# ==========================================
# 3. Training Logic: Double DQN + Dueling DQN ("Player Mode")
# ==========================================
def train_double_dueling_dqn_player_mode(env, episodes=500, batch_size=32, gamma=0.99, lr=1e-3, target_update_freq=10):
    """
    Training loop combining Dueling Architecture with Double DQN target calculation.
    Targeted for "Player Mode" where the player starts in a random position.
    """
    input_dim = env.observation_space.shape[0] if hasattr(env, 'observation_space') else 64
    output_dim = env.action_space.n if hasattr(env, 'action_space') else 4
    
    # Initialize Online and Target networks
    online_model = DuelingDQN(input_dim, output_dim)
    target_model = copy.deepcopy(online_model) # Target network starts as identical copy
    target_model.eval() # Target network does not track gradients
    
    optimizer = optim.Adam(online_model.parameters(), lr=lr)
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
            if random.random() < epsilon:
                action = random.randint(0, output_dim - 1)
            else:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    q_values = online_model(state_tensor)
                action = q_values.argmax().item()

            next_state, reward, done, _ = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if len(replay_buffer) >= batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                
                states = torch.FloatTensor(states)
                actions = torch.LongTensor(actions)
                rewards = torch.FloatTensor(rewards)
                next_states = torch.FloatTensor(next_states)
                dones = torch.FloatTensor(dones)

                # Online network predicts current Q(s, a)
                q_values = online_model(states)
                current_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

                # ========================================================
                # DOUBLE DQN TARGET CALCULATION
                # 1. Online Network selects the best action for the next state
                # 2. Target Network evaluates the Q-value of that selected action
                # ========================================================
                with torch.no_grad():
                    # 1. Action selection
                    best_next_actions = online_model(next_states).argmax(1)
                    
                    # 2. Action evaluation
                    next_target_q_values = target_model(next_states)
                    max_next_q = next_target_q_values.gather(1, best_next_actions.unsqueeze(1)).squeeze(1)
                    
                    # 3. Target calculation
                    target_q = rewards + gamma * max_next_q * (1 - dones)

                loss = loss_fn(current_q, target_q)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                episode_losses.append(loss.item())

        # Update Target Network periodically
        if episode % target_update_freq == 0:
            target_model.load_state_dict(online_model.state_dict())

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        rewards_history.append(total_reward)
        avg_loss = sum(episode_losses)/len(episode_losses) if episode_losses else 0.0
        loss_history.append(avg_loss)
        
        if episode % 50 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}, Loss: {avg_loss:.4f}")

    return online_model, rewards_history, loss_history

if __name__ == "__main__":
    from dummy_env import DummyGridWorld
    env = DummyGridWorld(mode='player')
    print("Starting HW3-2 Training (50 episodes)...")
    model, rewards, losses = train_double_dueling_dqn_player_mode(env, episodes=50, batch_size=16)
    print("HW3-2 Training Complete. Final Reward:", rewards[-1])
