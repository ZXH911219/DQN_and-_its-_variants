import numpy as np

class Space:
    def __init__(self, shape, n):
        self.shape = shape
        self.n = n
    def sample(self):
        return np.random.randint(0, self.n)

class RealGridWorld:
    def __init__(self, mode='static'):
        self.mode = mode
        self.action_space = Space((64,), 4)
        self.observation_space = Space((64,), 4)
        self.board_size = 4
        
        # Coordinates defined as (row, col) where (0,0) is top-left
        self.goal_pos = (0, 0)
        self.pit_pos = (0, 1)
        self.wall_pos = (1, 1)
        self.player_pos = (0, 3)
        
        self.max_steps = 30
        self.step_count = 0

    def reset(self):
        self.step_count = 0
        if self.mode == 'player':
            while True:
                r, c = np.random.randint(0, 4), np.random.randint(0, 4)
                if (r, c) not in [self.wall_pos, self.goal_pos, self.pit_pos]:
                    self.player_pos = (r, c)
                    break
        elif self.mode == 'random':
            positions = []
            while len(positions) < 4:
                r, c = np.random.randint(0, 4), np.random.randint(0, 4)
                if (r, c) not in positions:
                    positions.append((r, c))
            self.player_pos, self.goal_pos, self.pit_pos, self.wall_pos = positions
            
        return self._get_state()

    def _get_state(self):
        # 4x4x4 tensor flattened to 64
        # Channel 0: Player, 1: Goal, 2: Pit, 3: Wall
        state = np.zeros((4, 4, 4), dtype=np.float32)
        state[0, self.player_pos[0], self.player_pos[1]] = 1.0
        state[1, self.goal_pos[0], self.goal_pos[1]] = 1.0
        state[2, self.pit_pos[0], self.pit_pos[1]] = 1.0
        state[3, self.wall_pos[0], self.wall_pos[1]] = 1.0
        return state.flatten()

    def step(self, action):
        self.step_count += 1
        r, c = self.player_pos
        
        # 0: Up, 1: Down, 2: Left, 3: Right
        if action == 0:
            r = max(0, r - 1)
        elif action == 1:
            r = min(self.board_size - 1, r + 1)
        elif action == 2:
            c = max(0, c - 1)
        elif action == 3:
            c = min(self.board_size - 1, c + 1)
            
        # Move only if not hitting the wall
        if (r, c) != self.wall_pos:
            self.player_pos = (r, c)
            
        # Calculate rewards
        reward = -1.0 # step penalty
        done = False
        
        if self.player_pos == self.goal_pos:
            reward = 10.0
            done = True
        elif self.player_pos == self.pit_pos:
            reward = -10.0
            done = True
            
        if self.step_count >= self.max_steps:
            done = True
            
        return self._get_state(), reward, done, {}
