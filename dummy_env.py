import numpy as np

class DummyGridWorld:
    def __init__(self, mode='static'):
        self.mode = mode
        self.state_dim = 64
        self.action_dim = 4
        
        class Space:
            def __init__(self, shape, n):
                self.shape = shape
                self.n = n
            def sample(self):
                return np.random.randint(0, self.n)
                
        self.observation_space = Space((self.state_dim,), self.action_dim)
        self.action_space = Space((self.state_dim,), self.action_dim)
        self.step_count = 0
        self.max_steps = 15
        
    def reset(self):
        self.step_count = 0
        return np.random.randn(self.state_dim).astype(np.float32)
        
    def step(self, action):
        self.step_count += 1
        done = self.step_count >= self.max_steps
        
        # Give a positive reward if the agent takes action 0 (just to give it something to learn)
        reward = 1.0 if action == 0 else -0.1
        if done:
            reward += 10.0
            
        next_state = np.random.randn(self.state_dim).astype(np.float32)
        return next_state, reward, done, {}
