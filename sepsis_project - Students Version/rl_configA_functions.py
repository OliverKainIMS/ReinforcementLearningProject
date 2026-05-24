# --- IMPORTS ---

import numpy as np
import random

SEED = 42
np.random.seed(SEED)

from envs.env_setup import (
    ENV_ID, N_STATES, N_ACTIONS, STATE_SURVIVED, STATE_DIED,
    GAMMA, INTENSITY, SOFA_BIAS, LAM,
    make_sepsis_env,
)


# --- CONFIGURATION A: POLICY ITERATION ---

# 1. Build a deterministic policy evaluation function.

class EvalMetrics:
    """
    Stores evaluation statistics collected while testing a policy.

    This class tracks:
    - Episode returns (total reward per episode)
    - Episode lengths (number of steps per episode)
    - Survival outcomes (whether the episode achieved a positive return)

    Attributes
    ----------
    episode_returns: list[float]
        Total cumulative reward obtained in each evaluation episode.

    episode_lenghts: list[int]
        Number of environment steps taken in each episode.

    survival_flags: list[bool]
        Boolean indicators showing whether the episode was considered successful/ survived.
        An episode is marked as survived if its total return is greater than zero.

    """
    
    def __init__(self):
        """ Initialize empty metric storage containers."""
        self.episode_returns = []
        self.episode_lengths = []
        self.survival_flags = []

    def add(self, r, length):
        """ 
        Add the results of a completed evaluation episode.

        Parameters
        ----------
        r: float
            Total cumulative reward obtained in the episode.

        length: int
            Number of steps executed during the episode.
        """
        self.episode_returns.append(r)
        self.episode_lengths.append(length)
        self.survival_flags.append(r > 0)

    def summary(self):
        """ 
        Compute aggregate evaluation statistics.

        Returns
        -------
        dict
            Dictionary containing:
            - mean_return: float
                Average cumulative reward across episodes.
            - survival_rate: float
                Fraction of episodes with positive return.
            - mean_length: float
                Average episode length.
        """
        return {
            "mean_return": np.mean(self.episode_returns),
            "survival_rate": np.mean(self.survival_flags),
            "mean_length": np.mean(self.episode_lengths),
        }
    

def evaluate_policy(policy, n_episodes=1000, seed=SEED):
    """
    Evaluate a policy over multiple episodes in the sepsis environment.

    The function executes the provided policy in a fresh evaluation
    environment and records performance metrics such as episode return,
    episode length, and survival rate.

    Parameters
    ----------
    policy : array-like or dict
        Mapping from environment states to actions.
        The selected action is obtained using:
            action = policy[state]

    n_episodes : int, optional
        Number of evaluation episodes to run.
        Default is 1000.

    seed : int, optional
        Random seed used for reproducibility.
        Default is SEED.

    Returns
    -------
    EvalMetrics
        Object containing evaluation statistics collected across all episodes.

    Notes
    -----
    - A new random seed is sampled for each episode reset to ensure
      diverse trajectories during evaluation.
    - Episodes terminate when either:
        * terminated == True
        * truncated == True
    """

    env_eval = make_sepsis_env()
    metrics = EvalMetrics()

    np.random.seed(seed)

    for _ in range(n_episodes):

        # Reset environment with a random episode seed
        obs, _ = env_eval.reset(seed=np.random.randint(100000))

        done = False
        total_r = 0
        steps = 0

        while not done:

            # Select action from policy
            action = policy[obs]

            # Execute action in environment  
            obs, r, terminated, truncated, _ = env_eval.step(action)

            total_r += r
            steps += 1

            # Episode ends if terminated or truncated
            done = terminated or truncated

        # Store episode statistics
        metrics.add(total_r, steps)

    env_eval.close()

    return metrics

# 2. Implement the policy iteration algorithm.

def policy_evaluation(policy, P, R, gamma=GAMMA, theta=1e-8, max_iterations=1000):
    """
    Evaluate a fixed policy using iterative policy evaluation.

    This function computes the state-value function V(s) for a given
    policy by repeatedly applying the Bellman expectation update until
    convergence.

    Parameters
    ----------
    policy : np.ndarray
        Array of shape (N_STATES,) containing the action selected
        in each state.

    P : np.ndarray
        Transition probability matrix with shape:
            (N_STATES, N_ACTIONS, N_STATES)

        where:
            P[s, a, s'] = probability of transitioning
            from state s to state s' after taking action a.

    R : np.ndarray
        Reward matrix with shape:
            (N_STATES, N_ACTIONS, N_STATES)

        where:
            R[s, a, s'] = reward received when transitioning
            from state s to state s' using action a.

    gamma : float, optional
        Discount factor.
        Default is GAMMA.

    theta : float, optional
        Convergence tolerance threshold.
        Iteration stops when the maximum change in the value function
        falls below theta.
        Default is 1e-8.

    max_iterations : int, optional
        Maximum number of policy evaluation iterations.
        Default is 1000.

    Returns
    -------
    V : np.ndarray
        Estimated value function for the input policy.
        Shape: (N_STATES,)

    delta_history : list[float]
        History of maximum value-function updates across iterations.
        Useful for convergence analysis and visualization.
    """

    V = np.zeros(N_STATES)

    delta_history = []

    for iteration in range(max_iterations):

        V_new = np.zeros_like(V)

        for s in range(N_STATES):

            a = policy[s]

            V_new[s] = np.sum(P[s, a] * (R[s, a] + gamma * V))

        delta = np.max(np.abs(V_new - V))

        delta_history.append(delta)

        V = V_new

        if delta < theta:
            print(f'Policy evaluation converged in {iteration} iterations')
            break

    return V, delta_history

def policy_improvement(V, P, R, gamma=GAMMA):
    """
    Improve a policy greedily with respect to a value function.

    This function computes the action-value estimates Q(s, a) for
    every state-action pair and selects the action with the highest
    expected return.

    Parameters
    ----------
    V : np.ndarray
        State-value function estimate.
        Shape: (N_STATES,)

    P : np.ndarray
        Transition probability matrix with shape:
            (N_STATES, N_ACTIONS, N_STATES)

    R : np.ndarray
        Reward matrix with shape:
            (N_STATES, N_ACTIONS, N_STATES)

    gamma : float, optional
        Discount factor.
        Default is GAMMA.

    Returns
    -------
    policy : np.ndarray
        Improved deterministic policy.
        Shape: (N_STATES,)

        Each entry contains the action index selected for that state.
    """

    policy = np.zeros(N_STATES, dtype=int)

    for s in range(N_STATES):

        q_values = np.zeros(N_ACTIONS)

        for a in range(N_ACTIONS):

            q_values[a] = np.sum(P[s, a] * (R[s, a] + gamma * V))

        policy[s] = np.argmax(q_values)

    return policy

def policy_iteration(P, R, gamma=GAMMA):
    """
    Solve the MDP using the Policy Iteration algorithm.

    Policy Iteration alternates between:
    1. Policy Evaluation:
       Estimate the value function of the current policy.
    2. Policy Improvement:
       Construct a greedy policy with respect to the updated value function.

    The algorithm terminates once the policy no longer changes,
    indicating convergence to an optimal policy.

    Parameters
    ----------
    P : np.ndarray
        Transition probability matrix with shape:
            (N_STATES, N_ACTIONS, N_STATES)

    R : np.ndarray
        Reward matrix with shape:
            (N_STATES, N_ACTIONS, N_STATES)

    gamma : float, optional
        Discount factor.
        Default is GAMMA.

    Returns
    -------
    policy : np.ndarray
        Optimal policy found by policy iteration.
        Shape: (N_STATES,)

    V : np.ndarray
        Final state-value function corresponding to the optimal policy.
        Shape: (N_STATES,)

    metrics : dict
        Dictionary containing convergence statistics:
        - policy_changes : list[int]
            Number of states whose actions changed after each
            policy improvement step.
        - delta_history : list[float]
            History of value-function update magnitudes collected
            during policy evaluation.
        - iterations : int
            Total number of policy iteration cycles performed.

    """

    policy = np.zeros(N_STATES, dtype=int)

    stable = False
    iteration = 0

    policy_change_history = []
    delta_histories = []

    while not stable:

        iteration += 1

        old_policy = policy.copy()

        # Step 1: Policy Evaluation
        V, delta_history = policy_evaluation(policy, P, R, gamma)

        # Step 2: Policy Improvement
        new_policy = policy_improvement(V, P, R, gamma)

        # Count how many states change action
        changed_states = np.sum(old_policy != new_policy)

        policy_change_history.append(changed_states)

        delta_histories.extend(delta_history)

        # Check policy stability

        stable = np.array_equal(old_policy, new_policy)

        policy = new_policy

        print(
            f'PI Iteration {iteration} | '
            f'Changed states: {changed_states}'
        )

    metrics = {
        "policy_changes": policy_change_history,
        "delta_history": delta_histories,
        "iterations": iteration
    }

    return policy, V, metrics


# --- CONFIGURATION A: Q-LEARNING ---

# 1. Train a Q-learning agent.

def train_q_learning(n_episodes=10000, alpha=0.05, gamma=GAMMA, epsilon_start=1.0, epsilon_min=0.05, epsilon_decay=0.9995, q_init=0.0, 
                     seed=SEED):

    """
    Train a Q-learning agent on the sepsis environment using
    epsilon-greedy exploration.

    Parameters
    ----------
    n_episodes : int, default=10000
        Number of training episodes.

    alpha : float, default=0.05
        Learning rate controlling how strongly new information
        updates existing Q-values.

    gamma : float, default=GAMMA
        Discount factor for future rewards.

    epsilon_start : float, default=1.0
        Initial exploration probability for epsilon-greedy policy.

    epsilon_min : float, default=0.05
        Minimum exploration rate after decay.

    epsilon_decay : float, default=0.9995
        Multiplicative decay applied to epsilon after each episode.

    q_init : float, default=0.0
        Initial value assigned to all entries in the Q-table.

    seed : int, default=SEED
        Random seed for reproducibility.

    Returns
    -------
    dict
        Dictionary containing:

        - "Q" : np.ndarray
            Learned Q-table of shape (N_STATES, N_ACTIONS).

        - "returns" : list[float]
            Total reward obtained per episode.

        - "lengths" : list[int]
            Number of steps taken in each episode.

        - "survivals" : list[bool]
            Whether the episode ended with positive reward.

        - "epsilons" : list[float]
            Epsilon value used for each episode.

        - "td_errors" : list[float]
            Absolute temporal-difference errors recorded during training.

        - "q_changes" : list[float]
            Mean absolute Q-table change after each episode.

        - "exploration_ratio" : list[float]
            Fraction of exploratory actions taken per episode.

    Notes
    -----
    The agent follows an epsilon-greedy policy:
    - Explore with probability epsilon
    - Exploit the best-known action otherwise

    Epsilon decays after every episode until epsilon_min is reached.
    """

    env = make_sepsis_env()

    np.random.seed(seed)

    Q = np.ones((N_STATES, N_ACTIONS)) * q_init

    epsilon = epsilon_start

    returns = []
    lengths = []
    survivals = []
    epsilons = []

    td_error_history = []
    q_change_history = []
    exploration_history = []

    for ep in range(n_episodes):

        state, _ = env.reset(seed=np.random.randint(100000))

        done = False
        total_reward = 0
        steps = 0

        explore_count = 0
        exploit_count = 0

        Q_old = Q.copy()

        while not done:

            # epsilon-greedy
            if np.random.rand() < epsilon:
                action = np.random.randint(N_ACTIONS)
                explore_count += 1
            else:
                action = np.argmax(Q[state])
                exploit_count += 1

            next_state, reward, terminated, truncated, _ = env.step(action)

            done = terminated or truncated

            # Q-learning update
            best_next = np.argmax(Q[next_state])

            td_target = reward + gamma * Q[next_state, best_next]

            td_error = td_target - Q[state, action]

            Q[state, action] += alpha * td_error

            td_error_history.append(abs(td_error))

            state = next_state
            total_reward += reward
            steps += 1

        q_change = np.mean(np.abs(Q - Q_old))

        q_change_history.append(q_change)

        exploration_ratio = explore_count / max(1, explore_count + exploit_count)

        exploration_history.append(exploration_ratio)

        returns.append(total_reward)
        lengths.append(steps)
        survivals.append(total_reward > 0)
        epsilons.append(epsilon)

        # epsilon decay
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    env.close()

    return {
    "Q": Q,
    "returns": returns,
    "lengths": lengths,
    "survivals": survivals,
    "epsilons": epsilons,
    "td_errors": td_error_history,
    "q_changes": q_change_history,
    "exploration_ratio": exploration_history
}


# 2. Sample parameters for hyperparameter tuning.

def sample_params(param_grid):

    """
    Randomly sample a set of hyperparameters from a parameter grid.

    Parameters
    ----------
    param_grid : dict
        Dictionary containing lists of candidate values for each
        hyperparameter.

        Expected keys:
        - "alpha"
        - "epsilon_decay"
        - "epsilon_min"
        - "q_init"

    Returns
    -------
    dict
        Dictionary containing one randomly selected value for
        each hyperparameter.

    """

    return {
        "alpha": random.choice(param_grid["alpha"]),
        "epsilon_decay": random.choice(param_grid["epsilon_decay"]),
        "epsilon_min": random.choice(param_grid["epsilon_min"]),
        "q_init": random.choice(param_grid["q_init"]),
    }

# 3. Helper function to estimate training convergence.
# We are defining convergence as "when moving average return does not improve significantly for several consecutive episodes".

def q_convergence_episode(returns, window=100, patience = 5, threshold=0.02):
    
    """
    Estimate the convergence episode for Q-learning training
    using a moving average stability criterion.

    Parameters
    ----------
    returns : list[float] or np.ndarray
        Episode return values collected during training.

    window : int, default=100
        Window size used to compute the moving average of returns.

    patience : int, default=5
        Number of consecutive moving-average updates that must
        remain within the improvement threshold before convergence
        is declared.

    threshold : float, default=0.02
        Minimum required improvement between consecutive moving
        average values. Changes smaller than this threshold are
        considered stagnant.

    Returns
    -------
    int
        Estimated episode index where convergence occurred.
        If convergence is not detected, returns the total
        number of episodes.

    Notes
    -----
    The function computes a moving average over episode returns:
            MA_t = mean(returns[t-window : t])
    """

    ma = np.convolve(returns, np.ones(window)/window, mode='valid')
    
    stagnant = 0
    
    for i in range(1, len(ma)):
        if abs(ma[i] - ma[i-1]) < threshold:
            stagnant += 1
        else:
            stagnant = 0

        if stagnant >= patience:
            return i
            
    return len(returns)



# --- CONFIGURATION A: LEARNING CURVES ---

# 1. Helper function for moving average smoothing.

def moving_average(x, window=100):

    """
    Compute the moving average of a 1D sequence.

    Parameters
    ----------
    x : list[float] or np.ndarray
        Input sequence to smooth.

    window : int, default=100
        Size of the moving average window.

    Returns
    -------
    np.ndarray
        Smoothed sequence containing the moving average values.

    Notes
    -----
    - `mode='valid'` only returns values where the full averaging window overlaps with the input sequence.
    """
    
    return np.convolve(x, np.ones(window)/window, mode='valid')