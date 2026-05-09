# HW3-2 Enhanced DQN Variants: Double DQN and Dueling DQN

## 1. Double DQN

**The Problem with Naive DQN (Overestimation):**
In a basic DQN, the target Q-value is calculated using the formula:
`Target = Reward + Gamma * Max(Next Q(s', a'))`
Because we are using the `max` operator over noisy, imperfect estimates (especially early in training), the network tends to consistently *overestimate* the true action values. Since these overestimated values are used to update the network, the error propagates, causing training instability and sub-optimal policies.

**How Double DQN Improves It:**
Double DQN decouples the "selection" of the action from the "evaluation" of the action to prevent this overestimation.
1. **Action Selection:** The *Online Network* (which is actively training) decides which action is the best in the next state: `argmax Q_online(s', a)`.
2. **Action Evaluation:** The *Target Network* (which is slowly updated and stable) evaluates the value of that chosen action: `Q_target(s', argmax Q_online(s', a))`.

By separating these two steps, if the online network has a noisy spike in a Q-value for a particular action, it might select it, but the target network will provide a more grounded, realistic evaluation, significantly reducing the overestimation bias. This is highly effective in stochastic environments or the "Player Mode" where the start position is randomized.

---

## 2. Dueling DQN

**The Problem with Naive DQN (State Value vs. Action Advantage):**
In standard DQN, the final layer outputs the Q-value for every action. However, in many states, the choice of action doesn't actually matter much. For example, if a player is far away from the goal and there are no pits nearby, moving up or down might be equally okay, and the dominant factor is just the intrinsic value of being in that state. A standard DQN has to learn the exact Q-value for each action separately, which is inefficient.

**How Dueling DQN Improves It:**
Dueling DQN splits the network's final layers into two separate streams:
1. **Value Stream $V(s)$:** Estimates how good it is to be in a particular state, regardless of the action taken.
2. **Advantage Stream $A(s, a)$:** Estimates how much better taking a specific action is compared to the average action in that state.

These two streams are then aggregated at the final output layer:
`Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))`
*(Subtracting the mean helps the network stably identify the V and A components).*

**Why it performs better:**
By explicitly learning the state-value function $V(s)$, the network can learn which states are valuable (or dangerous) without having to experience every single action in those states. If moving into a state is inherently bad (e.g. right next to a pit), the Value stream learns this once, and it immediately applies to all actions leading to or from that state. This leads to faster convergence and a much better understanding of the environment, which is crucial in dynamic settings like "Player Mode".
