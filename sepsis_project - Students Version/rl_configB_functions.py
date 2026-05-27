"""
rl_configB_functions.py
=======================
Helper functions for Configuration B: Deep Q-Network (DQN) on the
Clinical ICU-Sepsis environment with continuous observations.

Contents
--------
  EvalMetricsB            – episode statistics tracker (extends Config A's EvalMetrics
                            with wrapper-specific failure-mode flags)
  evaluate_policy_b       – greedy evaluation of a DQN agent on make_clinical_env()
  QNetwork                – feedforward Q-network  (47 → hidden → 25)
  ReplayBuffer            – uniform circular experience replay buffer
  DQNAgent                – Double-DQN agent with target network & ε-greedy exploration
  train_dqn               – full training loop returning per-episode metrics
  sample_params_b         – random hyperparameter sampler for DQN grid search
  moving_average          – 1-D moving-average smoothing (shared plotting utility)
  dqn_convergence_episode – estimate the episode at which training converged
"""

import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

from envs.wrappers import make_clinical_env

# ─── Shared constants ─────────────────────────────────────────────────────────

SEED      = 42
N_OBS     = 47    # dimensionality of the continuous observation vector
N_ACTIONS = 25    # 5 vasopressor levels × 5 IV-fluid dose levels
GAMMA     = 1.0   # no time discounting, following the ICU-Sepsis paper convention


# ─── Evaluation helpers ───────────────────────────────────────────────────────

class EvalMetricsB:
    """
    Collects per-episode statistics for Config B policy evaluation.

    Extends the Config A EvalMetrics with wrapper-specific flags so that
    performance can be stratified by failure mode (noisy observations,
    missing features, acute events).

    Attributes
    ----------
    episode_returns : list[float]
        Total undiscounted return per episode.

    episode_lengths : list[int]
        Number of environment steps per episode.

    survival_flags : list[bool]
        True if the episode return was positive (patient survived).

    noisy_flags : list[bool]
        True if EpisodicNoisyObsEnv was active for that episode.

    missing_flags : list[bool]
        True if EpisodicMissingObsEnv was active for that episode.

    acute_flags : list[bool]
        True if AcuteEventEnv fired at least once during the episode.
    """

    def __init__(self):
        self.episode_returns = []
        self.episode_lengths = []
        self.survival_flags  = []
        self.noisy_flags     = []
        self.missing_flags   = []
        self.acute_flags     = []

    def add(self, r, length, noisy=False, missing=False, acute=False):
        """
        Record results of one completed episode.

        Parameters
        ----------
        r       : float — total undiscounted return
        length  : int   — number of steps taken
        noisy   : bool  — True if EpisodicNoisyObsEnv was active
        missing : bool  — True if EpisodicMissingObsEnv was active
        acute   : bool  — True if AcuteEventEnv fired at least once
        """
        self.episode_returns.append(r)
        self.episode_lengths.append(length)
        self.survival_flags.append(r > 0)
        self.noisy_flags.append(noisy)
        self.missing_flags.append(missing)
        self.acute_flags.append(acute)

    def summary(self):
        """
        Compute aggregate statistics over all recorded episodes.

        Returns
        -------
        dict
            - mean_return   : float — mean total return
            - survival_rate : float — fraction of survived episodes
            - mean_length   : float — mean episode length
        """
        return {
            "mean_return":   np.mean(self.episode_returns),
            "survival_rate": np.mean(self.survival_flags),
            "mean_length":   np.mean(self.episode_lengths),
        }


def evaluate_policy_b(agent, n_episodes=1000, seed=SEED, **env_kwargs):
    """
    Evaluate a trained DQN agent on the clinical ICU-Sepsis environment.

    The agent acts greedily (epsilon = 0) throughout evaluation.
    Wrapper-specific flags are recorded so performance can be broken down
    by failure mode (noisy observations, missing features, acute events).

    Parameters
    ----------
    agent       : DQNAgent
        Trained agent exposing select_action(obs, greedy=True).

    n_episodes  : int, optional
        Number of evaluation roll-outs. Default is 1000.

    seed        : int, optional
        Master RNG seed for reproducibility. Default is SEED.

    **env_kwargs :
        Forwarded to make_clinical_env() — use for sensitivity analysis
        (e.g. varying malfunction_prob, event_prob, etc.).

    Returns
    -------
    EvalMetricsB
        Object containing evaluation statistics for all episodes.
    """
    env = make_clinical_env(**env_kwargs)
    metrics = EvalMetricsB()
    np.random.seed(seed)

    for _ in range(n_episodes):
        obs, info = env.reset(seed=int(np.random.randint(100_000)))
        done       = False
        total_r    = 0.0
        steps      = 0
        ep_noisy   = info.get('noisy_episode', False)
        ep_missing = info.get('missing_features') is not None
        ep_acute   = False

        while not done:
            action = agent.select_action(obs, greedy=True)
            obs, r, te, tr, info = env.step(action)
            total_r += r
            steps   += 1
            done     = te or tr
            if info.get('acute_event', False):
                ep_acute = True

        metrics.add(total_r, steps, noisy=ep_noisy, missing=ep_missing, acute=ep_acute)

    env.close()
    return metrics


# ─── Neural Network ───────────────────────────────────────────────────────────

class QNetwork(nn.Module):
    """
    Feedforward Q-network mapping continuous observations to Q-values.

    Architecture: obs_dim → Linear → ReLU → Linear → ReLU → Linear (n_actions)

    The output layer has no activation — raw Q-value estimates are returned
    so that argmax gives the greedy action.

    Parameters
    ----------
    obs_dim   : int — input dimensionality (47 for Config B)
    n_actions : int — number of discrete actions (25)
    hidden1   : int — width of the first hidden layer
    hidden2   : int — width of the second hidden layer
    """

    def __init__(self, obs_dim=N_OBS, n_actions=N_ACTIONS, hidden1=128, hidden2=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, n_actions),
        )

    def forward(self, x):
        return self.net(x)


# ─── Experience Replay ────────────────────────────────────────────────────────

class ReplayBuffer:
    """
    Uniform experience replay buffer for DQN.

    Stores (obs, action, reward, next_obs, done) transitions in a circular
    buffer and provides random mini-batch sampling to break the temporal
    correlations that would otherwise bias gradient estimates.

    Parameters
    ----------
    capacity : int
        Maximum number of transitions to store. Older entries are
        evicted once capacity is reached. Default is 50,000.
    """

    def __init__(self, capacity=50_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done):
        """
        Append one transition to the buffer.

        Parameters
        ----------
        obs      : np.ndarray — current 47-dim observation
        action   : int        — action taken
        reward   : float      — reward received
        next_obs : np.ndarray — next 47-dim observation
        done     : float      — 1.0 if terminal, 0.0 otherwise
        """
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size):
        """
        Draw a random mini-batch without replacement.

        Parameters
        ----------
        batch_size : int — number of transitions to sample

        Returns
        -------
        tuple of np.ndarray
            (obs, actions, rewards, next_obs, dones), each of length batch_size.
        """
        batch = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return (
            np.array(obs,      dtype=np.float32),
            np.array(actions,  dtype=np.int64),
            np.array(rewards,  dtype=np.float32),
            np.array(next_obs, dtype=np.float32),
            np.array(dones,    dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ─── DQN Agent ────────────────────────────────────────────────────────────────

class DQNAgent:
    """
    Double Deep Q-Network agent for the discrete-action, continuous-observation
    ICU-Sepsis environment.

    Double DQN decouples action selection (online network) from action evaluation
    (target network), reducing the Q-value overestimation bias of vanilla DQN.

    Key components
    --------------
    online_net    : QNetwork trained at every gradient step.
    target_net    : QNetwork synced from online_net every `target_update` steps.
    replay_buffer : ReplayBuffer holding recent (s, a, r, s', done) transitions.
    epsilon       : current exploration probability, decayed after each episode.

    Parameters
    ----------
    obs_dim         : int   — observation dimensionality (47)
    n_actions       : int   — number of discrete actions (25)
    lr              : float — Adam learning rate
    gamma           : float — discount factor (1.0 for ICU-Sepsis)
    epsilon_start   : float — initial exploration probability
    epsilon_min     : float — minimum exploration probability after decay
    epsilon_decay   : float — multiplicative per-episode decay factor
    buffer_capacity : int   — replay buffer capacity
    batch_size      : int   — mini-batch size for each gradient update
    target_update   : int   — gradient steps between online→target syncs
    hidden1         : int   — first hidden layer width
    hidden2         : int   — second hidden layer width
    device          : str   — 'cpu' or 'cuda'
    """

    def __init__(
        self,
        obs_dim=N_OBS,
        n_actions=N_ACTIONS,
        lr=1e-3,
        gamma=GAMMA,
        epsilon_start=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        buffer_capacity=50_000,
        batch_size=64,
        target_update=100,
        hidden1=128,
        hidden2=128,
        device='cpu',
    ):
        self.n_actions     = n_actions
        self.gamma         = gamma
        self.epsilon       = epsilon_start
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size    = batch_size
        self.target_update = target_update
        self.device        = torch.device(device)

        self.online_net = QNetwork(obs_dim, n_actions, hidden1, hidden2).to(self.device)
        self.target_net = QNetwork(obs_dim, n_actions, hidden1, hidden2).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer     = optim.Adam(self.online_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        self.steps_done    = 0

    def select_action(self, obs, greedy=False):
        """
        Choose an action using epsilon-greedy exploration.

        Parameters
        ----------
        obs    : np.ndarray — current 47-dim observation
        greedy : bool       — if True, always exploit (used during evaluation)

        Returns
        -------
        int — action index in [0, 24]
        """
        if not greedy and np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_vals = self.online_net(obs_t)
        return int(q_vals.argmax(dim=1).item())

    def update(self):
        """
        Sample a mini-batch and perform one Double-DQN gradient step.

        The Double-DQN target is:
            y = r + γ · Q_target(s', argmax_{a'} Q_online(s', a'))

        This separates action selection (online network) from value estimation
        (target network), reducing Q-value overestimation.

        Returns
        -------
        float
            Huber loss for this update step.
            Returns 0.0 if the replay buffer is not yet large enough.

        Notes
        -----
        Gradient clipping (max_norm=10) stabilises training given the sparse,
        delayed reward signal of the ICU-Sepsis environment.
        The target network is synced with the online network every
        `self.target_update` gradient steps.
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        obs, actions, rewards, next_obs, dones = self.replay_buffer.sample(self.batch_size)

        obs_t      = torch.tensor(obs,      device=self.device)
        actions_t  = torch.tensor(actions,  device=self.device).unsqueeze(1)
        rewards_t  = torch.tensor(rewards,  device=self.device)
        next_obs_t = torch.tensor(next_obs, device=self.device)
        dones_t    = torch.tensor(dones,    device=self.device)

        # Current Q-values for the actions that were actually taken
        q_current = self.online_net(obs_t).gather(1, actions_t).squeeze(1)

        # Double-DQN bootstrap: online network selects the action, target evaluates it
        with torch.no_grad():
            next_actions = self.online_net(next_obs_t).argmax(dim=1, keepdim=True)
            q_next       = self.target_net(next_obs_t).gather(1, next_actions).squeeze(1)
            q_target     = rewards_t + self.gamma * q_next * (1.0 - dones_t)

        # Huber loss is more robust to outlier returns than MSE
        loss = nn.functional.smooth_l1_loss(q_current, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.steps_done += 1
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return float(loss.item())

    def decay_epsilon(self):
        """Apply one multiplicative epsilon decay step (called once per episode)."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# ─── Training Loop ────────────────────────────────────────────────────────────

def train_dqn(
    n_episodes=3000,
    lr=1e-3,
    gamma=GAMMA,
    epsilon_start=1.0,
    epsilon_min=0.05,
    epsilon_decay=0.995,
    buffer_capacity=50_000,
    batch_size=64,
    target_update=100,
    hidden1=128,
    hidden2=128,
    seed=SEED,
    device='cpu',
    verbose=False,
):
    """
    Train a Double-DQN agent on the clinical ICU-Sepsis environment.

    The agent interacts with make_clinical_env() (all three wrappers active),
    stores transitions in a replay buffer, and trains the Q-network at every
    environment step once the buffer has enough samples.

    Parameters
    ----------
    n_episodes      : int, default=3000
        Number of training episodes.

    lr              : float, default=1e-3
        Adam learning rate.

    gamma           : float, default=GAMMA
        Discount factor for future rewards.

    epsilon_start   : float, default=1.0
        Initial exploration probability for epsilon-greedy policy.

    epsilon_min     : float, default=0.05
        Minimum exploration rate after decay.

    epsilon_decay   : float, default=0.995
        Multiplicative decay applied to epsilon after each episode.

    buffer_capacity : int, default=50_000
        Maximum number of transitions stored in the replay buffer.

    batch_size      : int, default=64
        Mini-batch size for each gradient update step.

    target_update   : int, default=100
        Number of gradient steps between target-network synchronisations.

    hidden1         : int, default=128
        Width of the first hidden layer of the Q-network.

    hidden2         : int, default=128
        Width of the second hidden layer of the Q-network.

    seed            : int, default=SEED
        RNG seed applied to Python, NumPy, and PyTorch.

    device          : str, default='cpu'
        Torch device string ('cpu' or 'cuda').

    verbose         : bool, default=False
        If True, print a progress summary every 500 episodes.

    Returns
    -------
    dict
        'agent'       : DQNAgent    — trained agent ready for evaluation
        'returns'     : list[float] — per-episode total returns
        'lengths'     : list[int]   — per-episode step counts
        'survivals'   : list[bool]  — survival outcome per episode
        'losses'      : list[float] — mean Huber loss per episode
        'epsilons'    : list[float] — epsilon value used each episode
        'noisy_eps'   : list[bool]  — EpisodicNoisyObsEnv active flags
        'missing_eps' : list[bool]  — EpisodicMissingObsEnv active flags
        'acute_eps'   : list[bool]  — AcuteEventEnv fired flags
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    env = make_clinical_env()

    agent = DQNAgent(
        lr=lr, gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        buffer_capacity=buffer_capacity,
        batch_size=batch_size,
        target_update=target_update,
        hidden1=hidden1,
        hidden2=hidden2,
        device=device,
    )

    returns, lengths, survivals = [], [], []
    losses, epsilons            = [], []
    noisy_eps, missing_eps, acute_eps = [], [], []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=int(np.random.randint(100_000)))
        done       = False
        total_r    = 0.0
        steps      = 0
        ep_losses  = []
        ep_noisy   = info.get('noisy_episode', False)
        ep_missing = info.get('missing_features') is not None
        ep_acute   = False

        while not done:
            action = agent.select_action(obs)
            next_obs, r, te, tr, info = env.step(action)
            done = te or tr

            # 'done' is stored as float so the bootstrap target is zeroed out at
            # terminal transitions — there is no future value after episode end
            agent.replay_buffer.push(obs, action, r, next_obs, float(done))

            loss = agent.update()
            if loss > 0.0:
                ep_losses.append(loss)

            obs      = next_obs
            total_r += r
            steps   += 1

            if info.get('acute_event', False):
                ep_acute = True

        agent.decay_epsilon()

        returns.append(total_r)
        lengths.append(steps)
        survivals.append(total_r > 0)
        losses.append(np.mean(ep_losses) if ep_losses else 0.0)
        epsilons.append(agent.epsilon)
        noisy_eps.append(ep_noisy)
        missing_eps.append(ep_missing)
        acute_eps.append(ep_acute)

        if verbose and (ep + 1) % 500 == 0:
            recent_ret  = np.mean(returns[-100:])
            recent_surv = np.mean(survivals[-100:])
            print(
                f"Ep {ep+1:4d}/{n_episodes} | "
                f"Return {recent_ret:.3f} | "
                f"Survival {recent_surv:.2%} | "
                f"ε={agent.epsilon:.3f}"
            )

    env.close()

    return {
        'agent':       agent,
        'returns':     returns,
        'lengths':     lengths,
        'survivals':   survivals,
        'losses':      losses,
        'epsilons':    epsilons,
        'noisy_eps':   noisy_eps,
        'missing_eps': missing_eps,
        'acute_eps':   acute_eps,
    }


# ─── Hyperparameter Utilities ─────────────────────────────────────────────────

def sample_params_b(param_grid):
    """
    Randomly sample one DQN configuration from a hyperparameter grid.

    Parameters
    ----------
    param_grid : dict
        Each key maps to a list of candidate values.
        Typical keys: 'lr', 'epsilon_decay', 'epsilon_min',
                      'batch_size', 'hidden1', 'hidden2'

    Returns
    -------
    dict
        One randomly selected value per key.
    """
    return {k: random.choice(v) for k, v in param_grid.items()}


# ─── Plotting Utilities ───────────────────────────────────────────────────────

def moving_average(x, window=100):
    """
    Compute a simple moving average of a 1-D sequence.

    Uses 'valid' convolution, so the output has length max(0, len(x) - window + 1).

    Parameters
    ----------
    x      : list[float] or np.ndarray — input sequence to smooth
    window : int, default=100          — averaging window size

    Returns
    -------
    np.ndarray
        Smoothed sequence of length len(x) - window + 1.
    """
    return np.convolve(np.array(x, dtype=float), np.ones(window) / window, mode='valid')


def dqn_convergence_episode(returns, window=100, patience=5, threshold=0.02):
    """
    Estimate the episode at which DQN training converged.

    Convergence is declared when the moving-average return changes by less
    than `threshold` for `patience` consecutive steps.

    Parameters
    ----------
    returns   : list[float] or np.ndarray
        Per-episode returns collected during training.

    window    : int, default=100
        Moving average window size.

    patience  : int, default=5
        Consecutive stable windows required to declare convergence.

    threshold : float, default=0.02
        Minimum change between consecutive moving-average values that
        counts as meaningful improvement.

    Returns
    -------
    int
        Estimated convergence episode index.
        Returns len(returns) if convergence is never detected.
    """
    ma = moving_average(returns, window)
    stagnant = 0
    for i in range(1, len(ma)):
        if abs(ma[i] - ma[i - 1]) < threshold:
            stagnant += 1
        else:
            stagnant = 0
        if stagnant >= patience:
            return i
    return len(returns)
