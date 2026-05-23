# --- IMPORTS ---

import numpy as np

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
    metrics = rl_configA_functions.EvalMetrics()

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