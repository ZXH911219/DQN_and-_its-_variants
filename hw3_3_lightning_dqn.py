import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import DataLoader, IterableDataset
import numpy as np
import random
from collections import deque
import copy

# ==========================================
# 1. Experience Replay Buffer (Dataset)
# ==========================================
class RLDataset(IterableDataset):
    """
    Iterable Dataset for PyTorch Lightning to draw samples from Replay Buffer.
    """
    def __init__(self, buffer, batch_size):
        self.buffer = buffer
        self.batch_size = batch_size

    def __iter__(self):
        # Sample an infinite stream from the replay buffer
        while True:
            if len(self.buffer) < self.batch_size:
                yield None
            else:
                batch = random.sample(self.buffer, self.batch_size)
                state, action, reward, next_state, done = map(np.stack, zip(*batch))
                yield (torch.FloatTensor(state), 
                       torch.LongTensor(action), 
                       torch.FloatTensor(reward), 
                       torch.FloatTensor(next_state), 
                       torch.FloatTensor(done))

# ==========================================
# 2. PyTorch Lightning DQN Module
# ==========================================
class LightningDQN(pl.LightningModule):
    def __init__(self, input_dim, output_dim, env, hidden_dim=128, batch_size=32, lr=1e-3, gamma=0.99, target_update_freq=10):
        super(LightningDQN, self).__init__()
        self.save_hyperparameters(ignore=['env'])
        
        self.env = env
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.lr = lr
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        
        # Networks (Using Dueling architecture for stability in Random mode)
        self.feature_layer = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.value_stream = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.advantage_stream = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))
        
        # Target network
        self.target_feature_layer = copy.deepcopy(self.feature_layer)
        self.target_value_stream = copy.deepcopy(self.value_stream)
        self.target_advantage_stream = copy.deepcopy(self.advantage_stream)
        self.target_feature_layer.eval()
        self.target_value_stream.eval()
        self.target_advantage_stream.eval()

        self.replay_buffer = deque(maxlen=2000)
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        # Huber Loss (Smooth L1) - Training Tip 3: Robustness to outliers
        self.loss_fn = nn.HuberLoss()
        
        # History tracking
        self.rewards_history = []
        self.loss_history = []
        self.current_episode_reward = 0
        self.current_epoch_losses = []
        
        # Pre-fill buffer slightly
        self.populate_buffer(100)

    def forward(self, x):
        features = self.feature_layer(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))

    def target_forward(self, x):
        features = self.target_feature_layer(x)
        value = self.target_value_stream(features)
        advantage = self.target_advantage_stream(features)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))

    def populate_buffer(self, steps):
        state = self.env.reset()
        for _ in range(steps):
            action = self.env.action_space.sample() if hasattr(self.env, 'action_space') else random.randint(0, self.output_dim - 1)
            next_state, reward, done, _ = self.env.step(action)
            self.replay_buffer.append((state, action, reward, next_state, done))
            state = next_state if not done else self.env.reset()

    def play_step(self):
        # Step environment to collect data
        if not hasattr(self, 'current_state'):
            self.current_state = self.env.reset()
            
        if random.random() < self.epsilon:
            action = self.env.action_space.sample() if hasattr(self.env, 'action_space') else random.randint(0, self.output_dim - 1)
        else:
            state_tensor = torch.FloatTensor(self.current_state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self(state_tensor)
            action = q_values.argmax().item()

        next_state, reward, done, _ = self.env.step(action)
        self.replay_buffer.append((self.current_state, action, reward, next_state, done))
        self.current_episode_reward += reward

        if done:
            self.rewards_history.append(self.current_episode_reward)
            self.current_episode_reward = 0
            self.current_state = self.env.reset()
        else:
            self.current_state = next_state
    
    def training_step(self, batch, batch_idx):
        self.play_step() # Collect data during training
        
        if batch is None:
            return None # Skip if buffer is not full enough
        
        states, actions, rewards, next_states, dones = batch
        
        # Current Q Values
        q_values = self.forward(states)
        current_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q Values (Double DQN logic)
        with torch.no_grad():
            best_next_actions = self.forward(next_states).argmax(1)
            target_q_values = self.target_forward(next_states)
            max_next_q = target_q_values.gather(1, best_next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + self.gamma * max_next_q * (1 - dones)

        # Loss Calculation using Huber Loss
        loss = self.loss_fn(current_q, target_q)
        
        self.current_epoch_losses.append(loss.item())
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        if self.current_epoch_losses:
            self.loss_history.append(sum(self.current_epoch_losses) / len(self.current_epoch_losses))
            self.current_epoch_losses = []
        # Update target network periodically
        if self.current_epoch % self.target_update_freq == 0:
            self.target_feature_layer.load_state_dict(self.feature_layer.state_dict())
            self.target_value_stream.load_state_dict(self.value_stream.state_dict())
            self.target_advantage_stream.load_state_dict(self.advantage_stream.state_dict())
        
        # Decay Epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.log('epsilon', self.epsilon)

    def configure_optimizers(self):
        # Training Tip 2: Adam Optimizer + StepLR Scheduler
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        dataset = RLDataset(self.replay_buffer, self.batch_size)
        return DataLoader(dataset=dataset, batch_size=None) # batch_size=None because dataset yields batches


# ==========================================
# 3. Execution (Trainer with Training Tips)
# ==========================================
if __name__ == "__main__":
    from dummy_env import DummyGridWorld
    env = DummyGridWorld(mode='random')
    model = LightningDQN(input_dim=64, output_dim=4, env=env)
    print("Starting HW3-3 Training (10 epochs)...")
    trainer = pl.Trainer(max_epochs=10, gradient_clip_val=1.0, enable_progress_bar=False, enable_model_summary=False, logger=False)
    trainer.fit(model)
    print("HW3-3 Training Complete.")
