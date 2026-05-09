# HW3-1 Understanding Report: Naive DQN & Experience Replay

## 1. What is Naive DQN?
A Naive Deep Q-Network (DQN) represents the Q-value function using a neural network rather than a Q-table. For environments with large state spaces (like GridWorld represented by coordinates or pixel arrays), storing every possible state-action value in a table becomes unfeasible. Naive DQN takes the current state as input and outputs the predicted Q-values for all possible actions. The network is trained by minimizing the Mean Squared Error (MSE) between the predicted Q-value and the target Q-value: `Target = Reward + Gamma * Max(Next Q)`.

## 2. The Role of the Experience Replay Buffer
In traditional online reinforcement learning, the agent learns directly from consecutive experiences as it navigates the environment. However, this approach causes two major issues when combined with Neural Networks:
1. **High Correlation of Data:** Sequential states in a trajectory are highly correlated. Feeding highly correlated data to a neural network destabilizes training and leads to "catastrophic forgetting", where the network quickly forgets past states to fit the current states.
2. **Data Inefficiency:** Each experience is used only once to compute a gradient step and then thrown away.

**How Experience Replay Solves This:**
The `ReplayBuffer` stores the agent's experiences (State, Action, Reward, Next State, Done) up to a certain capacity (e.g., 1000 in this HW). During training, instead of learning from the immediate next step, the agent randomly samples a batch of experiences (e.g., 32) from the buffer.
* **Breaks Correlation:** Random sampling breaks the temporal correlation of consecutive samples, satisfying the i.i.d (independent and identically distributed) assumption needed for stable stochastic gradient descent.
* **Increases Efficiency:** Rare but highly rewarding experiences are kept in the buffer and can be sampled and learned from multiple times.

## 3. Application to "Static Mode" GridWorld
In the "static mode" GridWorld, the goal, pit, wall, and player start positions are fixed. The environment is entirely deterministic. Because the environment does not change, a Naive DQN with an Experience Buffer is usually sufficient to quickly converge and find the optimal path to the goal without needing advanced stabilization techniques like target networks or dueling architectures.
