import torch
import torch.nn as nn
import numpy as np
from collections import deque
import random

# ==========================================
# HW3-4 (Bonus): Rainbow DQN Key Components
# 
# Rainbow DQN combines 6 extensions of DQN:
# 1. Double DQN
# 2. Prioritized Experience Replay (PER)
# 3. Dueling Networks
# 4. Multi-step Learning
# 5. Distributional RL
# 6. Noisy Nets
#
# Below are snippets for the two most impactful components 
# for solving the Random Mode GridWorld: PER and Multi-step returns.
# ==========================================

# ==========================================
# 1. Prioritized Experience Replay (PER)
# ==========================================
class PrioritizedReplayBuffer:
    def __init__(self, capacity=1000, alpha=0.6):
        """
        Prioritized Replay Buffer implementation.
        Alpha determines how much prioritization is used 
        (0 = uniform random, 1 = full prioritization).
        """
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def push(self, state, action, reward, next_state, done):
        max_prio = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        
        # New experiences get max priority to guarantee they are sampled at least once
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        """
        Sample experiences based on priority.
        Beta is used for Importance Sampling (IS) weights to correct the bias 
        introduced by non-uniform sampling.
        """
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        # Calculate sampling probabilities: P(i) = p_i^alpha / sum(p_k^alpha)
        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        batch = [self.buffer[idx] for idx in indices]

        # Importance Sampling Weights: w_i = (N * P(i))^-beta / max(w)
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done, indices, weights

    def update_priorities(self, indices, td_errors, offset=1e-5):
        """
        Update the priorities of the sampled transitions based on the TD-Error.
        """
        for idx, error in zip(indices, td_errors):
            self.priorities[idx] = abs(error) + offset


# ==========================================
# 2. Multi-step Returns (N-step DQN)
# ==========================================
class MultiStepBuffer:
    def __init__(self, n_steps=3, gamma=0.99):
        """
        Buffer to accumulate N steps before pushing to the main Replay Buffer.
        R_t = r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + ... + gamma^n * max_Q(S_{t+n}, a)
        """
        self.n_steps = n_steps
        self.gamma = gamma
        self.n_step_buffer = deque(maxlen=n_steps)

    def process_step(self, state, action, reward, next_state, done):
        """
        Add a step and return the n-step transition if buffer is full.
        """
        self.n_step_buffer.append((state, action, reward, next_state, done))
        
        if len(self.n_step_buffer) < self.n_steps:
            return None # Not enough steps yet
            
        # Calculate n-step reward
        n_step_reward = 0
        for idx, transition in enumerate(self.n_step_buffer):
            _, _, r, _, d = transition
            n_step_reward += (self.gamma ** idx) * r
            if d:
                break
                
        # Return: First State, First Action, N-step Reward, N-th Next State, Done flag
        first_state, first_action, _, _, _ = self.n_step_buffer[0]
        last_next_state, last_done = self.n_step_buffer[-1][3], self.n_step_buffer[-1][4]
        
        return first_state, first_action, n_step_reward, last_next_state, last_done

# ==========================================
# 3. How to use them together in Training Loop
# ==========================================
def rainbow_training_step_example(batch, model, target_model, optimizer, gamma=0.99, n_steps=3):
    """
    Demonstration of how the Loss calculation changes.
    """
    states, actions, rewards, next_states, dones, indices, weights = batch
    
    # 1. Online Network predicts current Q
    q_values = model(states)
    current_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    
    # 2. Target Network calculates Multi-step target
    with torch.no_grad():
        best_actions = model(next_states).argmax(1) # Double DQN logic
        next_q_values = target_model(next_states).gather(1, best_actions.unsqueeze(1)).squeeze(1)
        # Gamma is raised to the power of n_steps for the target Q
        target_q = rewards + (gamma ** n_steps) * next_q_values * (1 - dones)
        
    # 3. TD Errors for Priority Update
    td_errors = target_q - current_q
    
    # 4. Importance Sampling Loss 
    # Use the weights returned by PER buffer to scale the MSE/Huber Loss
    loss_fn = nn.MSELoss(reduction='none') # Don't mean the loss yet
    loss = (weights * loss_fn(current_q, target_q)).mean()
    
    optimizer.zero_grad()
    loss.backward()
    
    # Optional: Gradient Clipping
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    return loss, td_errors.detach().numpy()

def train_rainbow_dqn_random_mode(env, episodes=200, batch_size=32, gamma=0.99, lr=1e-3, target_update_freq=10):
    input_dim = env.observation_space.shape[0] if hasattr(env, 'observation_space') else 64
    output_dim = env.action_space.n if hasattr(env, 'action_space') else 4
    
    # We use a simple Dueling DQN architecture for the model
    # (Reusing DuelingDQN from hw3_2_double_dueling_dqn)
    import copy
    from hw3_2_double_dueling_dqn import DuelingDQN
    import torch.optim as optim
    
    model = DuelingDQN(input_dim, output_dim)
    target_model = copy.deepcopy(model)
    target_model.eval()
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    per_buffer = PrioritizedReplayBuffer(capacity=1000)
    n_step_buffer = MultiStepBuffer(n_steps=3, gamma=gamma)
    
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
                    q_values = model(state_tensor)
                action = q_values.argmax().item()

            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            
            # Process n-step
            n_step_transition = n_step_buffer.process_step(state, action, reward, next_state, done)
            if n_step_transition is not None:
                per_buffer.push(*n_step_transition)
                
            state = next_state

            if len(per_buffer.buffer) >= batch_size:
                s, a, r, ns, d, indices, weights = per_buffer.sample(batch_size)
                
                s = torch.FloatTensor(s)
                a = torch.LongTensor(a)
                r = torch.FloatTensor(r)
                ns = torch.FloatTensor(ns)
                d = torch.FloatTensor(d)
                weights = torch.FloatTensor(weights)
                
                batch = (s, a, r, ns, d, indices, weights)
                
                loss, td_errors = rainbow_training_step_example(batch, model, target_model, optimizer, gamma, n_steps=3)
                per_buffer.update_priorities(indices, td_errors)
                
                episode_losses.append(loss.item())

        if episode % target_update_freq == 0:
            target_model.load_state_dict(model.state_dict())

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        rewards_history.append(total_reward)
        avg_loss = sum(episode_losses)/len(episode_losses) if episode_losses else 0.0
        loss_history.append(avg_loss)
        
        if episode % 50 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}, Loss: {avg_loss:.4f}")

    return model, rewards_history, loss_history

if __name__ == "__main__":
    from dummy_env import DummyGridWorld
    env = DummyGridWorld(mode='random')
    print("Starting HW3-4 Training (50 episodes)...")
    model, rewards, losses = train_rainbow_dqn_random_mode(env, episodes=50, batch_size=16)
    print("HW3-4 Training Complete. Final Reward:", rewards[-1])
