"""
rl_configB_functions.py
=======================
Helper functions for Configuration B: DQN / Double DQN / PPO on the
Clinical ICU-Sepsis environment with continuous observations.

Contents
--------
  EvalMetricsB              - episode statistics tracker
  evaluate_policy_b         - greedy evaluation on make_clinical_env()
  QNetwork                  - feedforward Q-network (47 -> hidden -> 25)
  ActorCritic               - shared-trunk actor-critic network for PPO
  ReplayBuffer              - uniform circular experience replay buffer
  DQNAgent                  - DQN / Double-DQN agent (double=True for Double DQN)
  PPOAgent                  - PPO agent with GAE
  train_dqn                 - DQN / Double-DQN training loop
  train_ppo                 - PPO training loop
  run_optuna                - Optuna hyperparameter search for all three algorithms
  moving_average            - causal moving average (preserves sequence length)
  dqn_convergence_episode   - estimate convergence episode from returns
  plot_return_curves        - comparison learning-curve plot (return)
  plot_survival_curves      - comparison learning-curve plot (survival rate)

Assignment notes
----------------
  Double DQN are off-policy value-based methods. Their two key knobs
  are the **replay buffer size** (buffer_size) and the **target-network update
  frequency** (target_update_freq). These are explicit parameters in train_dqn
  and are included in the Optuna search space.

  PPO is an on-policy policy-gradient method. It has no replay buffer and no
  target network; its key knobs are rollout_length, clip_eps and gae_lambda.
"""

import os
import numpy as np
import pandas as pd
import random
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt

from envs.wrappers import make_clinical_env

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _HAS_OPTUNA = True
except ImportError:
    optuna = None
    _HAS_OPTUNA = False

# --- Shared constants ---

SEED      = 42
N_OBS     = 47    # dimensionality of the continuous observation vector
N_ACTIONS = 25    # 5 vasopressor levels x 5 IV-fluid dose levels
GAMMA     = 1.0   # no time discounting, following the ICU-Sepsis paper convention

# --- Helper Function ---
def get_default_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

# --- Evaluation helpers ---

class EvalMetricsB:
    """
    Collects per-episode statistics for Config B policy evaluation.

    Extends the Config A EvalMetrics with wrapper-specific flags so that
    performance can be stratified by failure mode (noisy observations,
    missing features, acute events).
    """

    def __init__(self):
        self.episode_returns = []
        self.episode_lengths = []
        self.survival_flags  = []
        self.noisy_flags     = []
        self.missing_flags   = []
        self.acute_flags     = []

    def add(self, r, length, noisy=False, missing=False, acute=False):
        self.episode_returns.append(r)
        self.episode_lengths.append(length)
        self.survival_flags.append(r > 0)
        self.noisy_flags.append(noisy)
        self.missing_flags.append(missing)
        self.acute_flags.append(acute)

    def summary(self):
        return {
            "mean_return":   np.mean(self.episode_returns),
            "survival_rate": np.mean(self.survival_flags),
            "mean_length":   np.mean(self.episode_lengths),
        }


def evaluate_policy_b(agent, n_episodes=1000, seed=SEED, **env_kwargs):
    """
    Evaluate a trained agent (DQN, Double DQN or PPO) on make_clinical_env().

    All three algorithm types expose select_action(obs, greedy=True), so this
    function works uniformly for DQNAgent and PPOAgent.

    Parameters
    ----------
    agent       : DQNAgent or PPOAgent
    n_episodes  : int
    seed        : int
    **env_kwargs : forwarded to make_clinical_env() for sensitivity analysis.

    Returns
    -------
    EvalMetricsB
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


# --- Neural Networks ---

class QNetwork(nn.Module):
    """
    Feedforward Q-network: obs (47) -> hidden -> ReLU -> ... -> Q-values (25).

    Parameters
    ----------
    obs_dim   : int  – input dimensionality (47 for Config B)
    n_actions : int  – number of discrete actions (25)
    hidden1   : int  – width of the first hidden layer
    hidden2   : int  – width of the second hidden layer
    """

    def __init__(self, obs_dim=N_OBS, n_actions=N_ACTIONS, hidden1=128, hidden2=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden1),
            nn.LayerNorm(hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.LayerNorm(hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class ActorCritic(nn.Module):
    """
    Shared-trunk actor-critic for PPO.

    A common feature trunk feeds two separate heads:
      * policy head -> action logits (25) for Categorical distribution
      * value  head -> scalar state value estimate

    Tanh activations are preferred for the PPO trunk as they bound the
    feature magnitudes and produce smoother advantage estimates.

    Parameters
    ----------
    obs_dim   : int
    n_actions : int
    hidden1   : int
    hidden2   : int
    """

    def __init__(self, obs_dim=N_OBS, n_actions=N_ACTIONS, hidden1=128, hidden2=128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden1),
            nn.LayerNorm(hidden1),
            nn.Tanh(),
            nn.Linear(hidden1, hidden2),
            nn.LayerNorm(hidden2),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden2, n_actions)
        self.value_head  = nn.Linear(hidden2, 1)

    def forward(self, x):
        z = self.trunk(x)
        return self.policy_head(z), self.value_head(z).squeeze(-1)


# --- Experience Replay ---

class ReplayBuffer:
    """
    Uniform circular experience-replay buffer for DQN / Double DQN.

    Stores (obs, action, reward, next_obs, done) transitions and provides
    random minibatch sampling to break temporal correlations in the training
    data, which is essential for stable convergence of value-based methods.

    Parameters
    ----------
    capacity : int
        Maximum number of transitions stored (the buffer_size hyperparameter).
        Older entries are evicted once capacity is reached.
    """

    def __init__(self, capacity=50_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size):
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


# --- DQN / Double DQN Agent ---

class DQNAgent:
    """
    DQN or Double DQN agent for the discrete-action, continuous-observation
    ICU-Sepsis environment.

    Set ``double=True`` for Double DQN, which decouples action *selection*
    (online network) from action *evaluation* (target network) to reduce the
    Q-value overestimation bias present in vanilla DQN.

    Assignment focus hyperparameters
    ---------------------------------
    buffer_size        : replay buffer capacity - a too-small buffer causes
                         catastrophic forgetting; a too-large one slows learning
                         because old transitions stay in the buffer too long.

    Parameters
    ----------
    double : bool
        If True, use the Double DQN bootstrap target.
    """

    def __init__(
        self,
        obs_dim=N_OBS,
        n_actions=N_ACTIONS,
        lr=1e-3,
        gamma=GAMMA,
        epsilon_start=1.0,
        epsilon_min=0.05,
        buffer_size=50_000,
        batch_size=64,
        tau=0.005,
        hidden1=128,
        hidden2=128,
        double=False,
        device=None,
    ):
        self.n_actions          = n_actions
        self.gamma              = gamma
        self.epsilon            = epsilon_start
        self.epsilon_min        = epsilon_min
        self.batch_size         = batch_size
        self.tau                = tau
        self.buffer_size        = buffer_size
        self.double             = double
        self.device             = torch.device(device if device is not None else get_default_device())

        self.online_net = QNetwork(obs_dim, n_actions, hidden1, hidden2).to(self.device)
        self.target_net = QNetwork(obs_dim, n_actions, hidden1, hidden2).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer     = optim.Adam(self.online_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.steps_done    = 0

    def select_action(self, obs, greedy=False):
        """
        Epsilon-greedy action selection.

        Parameters
        ----------
        greedy : bool  - if True (evaluation), always exploit.
        """
        if not greedy and np.random.rand() < self.epsilon:
            return int(np.random.randint(self.n_actions))
        obs_t = torch.tensor(
            np.array(obs, dtype=np.float32), device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            q_vals = self.online_net(obs_t)
        return int(q_vals.argmax(dim=1).item())

    def update(self):
        """
        Sample a minibatch and perform one gradient step.

        DQN target (vanilla):
            y = r + γ · max_{a'} Q_target(s', a')

        Double DQN target:
            y = r + γ · Q_target(s', argmax_{a'} Q_online(s', a'))

        Returns 0.0 if the buffer is not yet large enough to sample.
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        obs, actions, rewards, next_obs, dones = self.replay_buffer.sample(self.batch_size)

        obs_t      = torch.tensor(obs,      device=self.device)
        actions_t  = torch.tensor(actions,  device=self.device).unsqueeze(1)
        rewards_t  = torch.tensor(rewards,  device=self.device)
        next_obs_t = torch.tensor(next_obs, device=self.device)
        dones_t    = torch.tensor(dones,    device=self.device)

        q_current = self.online_net(obs_t).gather(1, actions_t).squeeze(1)

        with torch.no_grad():
            if self.double:
                next_actions = self.online_net(next_obs_t).argmax(dim=1, keepdim=True)
                q_next = self.target_net(next_obs_t).gather(1, next_actions).squeeze(1)
            else:
                q_next = self.target_net(next_obs_t).max(dim=1).values
            q_target = rewards_t + self.gamma * q_next * (1.0 - dones_t)

        loss = nn.functional.smooth_l1_loss(q_current, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Soft (Polyak) update after every gradient step.
        # target <- tau * online + (1 - tau) * target
        with torch.no_grad():
            for tp, op in zip(self.target_net.parameters(), self.online_net.parameters()):
                tp.copy_(self.tau * op + (1.0 - self.tau) * tp)

        return float(loss.item())


# --- PPO Agent ---

class PPOAgent:
    """
    Proximal Policy Optimisation with clipped objective and GAE.

    Unlike DQN / Double DQN, PPO is **on-policy**: there is no replay buffer
    and no target network. It collects fresh rollouts, computes generalised
    advantage estimates (GAE), and runs several minibatch update epochs over
    the collected data before discarding it.

    Key hyperparameters
    -------------------
    rollout_length  : steps collected per policy update
    update_epochs   : passes over each rollout
    clip_eps        : PPO clipping parameter (keeps the update conservative)
    gae_lambda      : GAE smoothing (λ=1 -> full Monte-Carlo returns,
                      λ=0 -> pure TD, λ≈0.95 is a common sweet spot)
    entropy_coef    : entropy bonus (encourages exploration)
    """

    def __init__(
        self,
        obs_dim=N_OBS,
        n_actions=N_ACTIONS,
        lr=3e-4,
        gamma=GAMMA,
        gae_lambda=0.95,
        clip_eps=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        rollout_length=1024,
        update_epochs=10,
        minibatch_size=128,
        hidden1=128,
        hidden2=128,
        max_grad_norm=0.5,
        device=None,
    ):
        self.gamma          = gamma
        self.gae_lambda     = gae_lambda
        self.clip_eps       = clip_eps
        self.entropy_coef   = entropy_coef
        self.value_coef     = value_coef
        self.rollout_length = rollout_length
        self.update_epochs  = update_epochs
        self.minibatch_size = minibatch_size
        self.max_grad_norm  = max_grad_norm
        self.device         = torch.device(device if device is not None else get_default_device())

        self.net       = ActorCritic(obs_dim, n_actions, hidden1, hidden2).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

    def act(self, obs):
        """
        Sample an action for on-policy data collection.

        Returns
        -------
        (int, float, float) – (action, log_prob, value_estimate)
        """
        obs_t = torch.tensor(
            np.array(obs, dtype=np.float32), device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            logits, value = self.net(obs_t)
            dist     = torch.distributions.Categorical(logits=logits)
            action   = dist.sample()
            log_prob = dist.log_prob(action)
        return int(action.item()), float(log_prob.item()), float(value.item())

    def select_action(self, obs, greedy=True):
        """Greedy or stochastic action selection for evaluation."""
        obs_t = torch.tensor(
            np.array(obs, dtype=np.float32), device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            logits, _ = self.net(obs_t)
            if greedy:
                return int(logits.argmax(dim=1).item())
            return int(torch.distributions.Categorical(logits=logits).sample().item())

    def value(self, obs):
        """Critic value estimate for bootstrapping at rollout boundaries."""
        obs_t = torch.tensor(
            np.array(obs, dtype=np.float32), device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            _, v = self.net(obs_t)
        return float(v.item())

    def update(self, b_obs, b_actions, b_log_probs, b_returns, b_advantages):
        """
        PPO clipped-objective update over one collected rollout.

        Advantage normalisation (zero mean, unit std) is applied before
        the update to reduce variance and stabilise training.
        """
        obs_t  = torch.tensor(b_obs,        dtype=torch.float32, device=self.device)
        acts_t = torch.tensor(b_actions,    dtype=torch.int64,   device=self.device)
        old_lp = torch.tensor(b_log_probs,  dtype=torch.float32, device=self.device)
        rets_t = torch.tensor(b_returns,    dtype=torch.float32, device=self.device)
        adv_t  = torch.tensor(b_advantages, dtype=torch.float32, device=self.device)
        adv_t  = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        n   = obs_t.shape[0]
        idx = np.arange(n)
        p_losses, v_losses, entropies, clip_fracs = [], [], [], []

        for _ in range(self.update_epochs):
            np.random.shuffle(idx)
            for start in range(0, n, self.minibatch_size):
                mb   = idx[start: start + self.minibatch_size]
                mb_t = torch.as_tensor(mb, dtype=torch.int64, device=self.device)

                logits, values = self.net(obs_t[mb_t])
                dist    = torch.distributions.Categorical(logits=logits)
                new_lp  = dist.log_prob(acts_t[mb_t])
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_lp - old_lp[mb_t])
                surr1 = ratio * adv_t[mb_t]
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_t[mb_t]
                p_loss = -torch.min(surr1, surr2).mean()
                v_loss = nn.functional.mse_loss(values, rets_t[mb_t])
                loss   = p_loss + self.value_coef * v_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    clip_fracs.append(float(
                        ((ratio - 1.0).abs() > self.clip_eps).float().mean()
                    ))
                p_losses.append(float(p_loss))
                v_losses.append(float(v_loss))
                entropies.append(float(entropy))

        return {
            'policy_loss':   float(np.mean(p_losses)),
            'value_loss':    float(np.mean(v_losses)),
            'entropy':       float(np.mean(entropies)),
            'clip_fraction': float(np.mean(clip_fracs)),
        }


# --- Training Loops ---

def train_dqn(
    n_episodes=20000,
    double=False,
    lr=1e-3,
    gamma=GAMMA,
    epsilon_start=1.0,
    epsilon_min=0.05,
    exploration_fraction=0.05,
    buffer_size=50000,
    batch_size=64,
    gradient_steps=1,
    hidden1=128,
    hidden2=128,
    learning_starts=1000,
    seed=SEED,
    device=None,
    checkpoint_every=None,
    checkpoint_eval_episodes=1000,
    checkpoint_eval_seeds=None,
    checkpoint_dir=None,
    checkpoint_prefix="dqn",
    verbose=False,
    log_every=1000,
):
    """
    Train a DQN or Double DQN agent on the clinical ICU-Sepsis environment.

    Returns
    -------
    dict
        'agent', 'returns', 'survivals', 'lengths', 'losses', 'epsilons',
        'noisy_eps', 'missing_eps', 'acute_eps', 'ppo_updates',
        'best_checkpoint', 'best_checkpoint_episode',
        'best_checkpoint_score', 'best_checkpoint_path'
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    env       = make_clinical_env()
    algo_name = 'Double DQN' if double else 'DQN'

    agent = DQNAgent(
        lr=lr, gamma=gamma,
        epsilon_start=epsilon_start, epsilon_min=epsilon_min,
        buffer_size=buffer_size, batch_size=batch_size,
        hidden1=hidden1, hidden2=hidden2,
        double=double, device=device,
    )

    # Linear epsilon schedule: decay over the first exploration_fraction of episodes.
    decay_episodes = max(1, int(exploration_fraction * n_episodes))

    returns, lengths, survivals = [], [], []
    losses, epsilons            = [], []
    noisy_eps, missing_eps, acute_eps = [], [], []
    total_steps = 0

    # Initialize checkpoint evaluation tracking variables
    if checkpoint_eval_seeds is None:
        checkpoint_eval_seeds = [seed]

    best_checkpoint = None
    best_checkpoint_score = (-np.inf, -np.inf)
    best_checkpoint_episode = None
    best_checkpoint_path = None

    for ep in range(n_episodes):
        # Linear epsilon decay (clipped at epsilon_min)
        agent.epsilon = max(
            epsilon_min,
            epsilon_start - (epsilon_start - epsilon_min) * ep / decay_episodes,
        )

        obs, info  = env.reset(seed=int(np.random.randint(100_000)))
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
            agent.replay_buffer.push(obs, action, r, next_obs, float(done))
            total_steps += 1

            if total_steps >= learning_starts:
                for _ in range(gradient_steps):
                    loss = agent.update()
                    if loss > 0.0:
                        ep_losses.append(loss)

            obs      = next_obs
            total_r += r
            steps   += 1
            if info.get('acute_event', False):
                ep_acute = True

        returns.append(total_r)
        lengths.append(steps)
        survivals.append(total_r > 0)
        losses.append(float(np.mean(ep_losses)) if ep_losses else 0.0)
        epsilons.append(agent.epsilon)
        noisy_eps.append(ep_noisy)
        missing_eps.append(ep_missing)
        acute_eps.append(ep_acute)

        if verbose and (ep + 1) % log_every == 0:
            recent_ret  = np.mean(returns[-log_every:])
            recent_surv = np.mean(survivals[-log_every:])
            print(
                f"[{algo_name}] Ep {ep+1:6d}/{n_episodes} | "
                f"Return {recent_ret:.3f} | Survival {recent_surv:.2%} | "
                f"ε={agent.epsilon:.4f} | buffer={len(agent.replay_buffer):6d}"
            )

        # It performs a checkpoint if checkpointing is enabled and if we've reached a multiple checkpoint_every episode
        # or if we've reached the last episode of training.
        should_checkpoint = (checkpoint_every is not None and ((ep + 1) % checkpoint_every == 0 or (ep + 1) == n_episodes))

        if should_checkpoint:
            # Preserve RNG states because evaluation resets np.random internally
            np_state = np.random.get_state()
            random_state = random.getstate()
            torch_state = torch.random.get_rng_state()
            cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

            # Perform evaluation with the current policy
            eval_result = evaluate_policy_multiseed(
                agent,
                eval_seeds=checkpoint_eval_seeds,
                n_episodes=checkpoint_eval_episodes,
            )
            summary = eval_result["summary"]

            # Restore RNG states to avoid affecting training
            np.random.set_state(np_state)
            random.setstate(random_state)
            torch.random.set_rng_state(torch_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)

            score = (summary["survival_rate"], summary["mean_return"])

            if score > best_checkpoint_score: #priority 1: survival rate, priority 2: mean return
                best_checkpoint_score = score
                best_checkpoint_episode = ep + 1

                best_checkpoint = {
                    "algo": algo_name,
                    "episode": best_checkpoint_episode,
                    "score": best_checkpoint_score,
                    "summary": summary,
                    "online_net": {
                        k: v.detach().cpu().clone()
                        for k, v in agent.online_net.state_dict().items()
                    },
                    "target_net": {
                        k: v.detach().cpu().clone()
                        for k, v in agent.target_net.state_dict().items()
                    },
                    "epsilon": agent.epsilon,
                    "buffer_size": buffer_size,
                    "batch_size": batch_size,
                    "gradient_steps": gradient_steps,
                    "hidden1": hidden1,
                    "hidden2": hidden2,
                    "lr": lr,
                    "gamma": gamma,
                    "tau": agent.tau,
                    "double": double,
                }

                # Save the best checkpoint to disk if checkpoint_dir is specified
                if checkpoint_dir is not None:
                    # Ensure the checkpoint directory exists, if not, create it
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    best_checkpoint_path = os.path.join(
                        checkpoint_dir,
                        f"{checkpoint_prefix}_best.pt"
                    )
                    torch.save(best_checkpoint, best_checkpoint_path)

                # Print checkpoint info if verbose
                if verbose:
                    print(
                        f"[{algo_name}] New best checkpoint at ep {ep + 1}: "
                        f"survival={score[0]:.2%} | return={score[1]:.4f}"
                    )

    env.close()

    # Load the best checkpoint's weights into the agent before returning, so that the returned agent is the best one found during training!
    if best_checkpoint is not None:
        agent.online_net.load_state_dict(best_checkpoint["online_net"])
        agent.target_net.load_state_dict(best_checkpoint["target_net"])
        agent.online_net.to(agent.device)
        agent.target_net.to(agent.device)
        agent.epsilon = best_checkpoint["epsilon"]

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
        'best_checkpoint': best_checkpoint,
        'best_checkpoint_episode': best_checkpoint_episode,
        'best_checkpoint_score': best_checkpoint_score,
        'best_checkpoint_path': best_checkpoint_path,
    }


def train_ppo(
    n_episodes=20000,
    lr=3e-4,
    gamma=GAMMA,
    gae_lambda=0.95,
    clip_eps=0.2,
    entropy_coef=0.01,
    value_coef=0.5,
    rollout_length=1024,
    update_epochs=10,
    minibatch_size=128,
    hidden1=128,
    hidden2=128,
    seed=SEED,
    device=None,
    checkpoint_every=None,
    checkpoint_eval_episodes=1000,
    checkpoint_eval_seeds=None,
    checkpoint_dir=None,
    checkpoint_prefix="ppo",
    verbose=False,
    log_every=1000,
):
    """
    Train a PPO agent on the clinical ICU-Sepsis environment.

    PPO is on-policy: no replay buffer and no target network. It collects
    rollout_length fresh steps, computes GAE advantages, and then runs
    update_epochs passes of minibatch updates before discarding the data.

    Returns
    -------
    dict
        'agent', 'returns', 'survivals', 'lengths', 'losses', 'epsilons' (NaN),
        'noisy_eps', 'missing_eps', 'acute_eps', 'ppo_updates',
        'best_checkpoint', 'best_checkpoint_episode',
        'best_checkpoint_score', 'best_checkpoint_path'
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    env   = make_clinical_env()
    agent = PPOAgent(
        lr=lr, gamma=gamma, gae_lambda=gae_lambda,
        clip_eps=clip_eps, entropy_coef=entropy_coef, value_coef=value_coef,
        rollout_length=rollout_length, update_epochs=update_epochs,
        minibatch_size=minibatch_size,
        hidden1=hidden1, hidden2=hidden2, device=device,
    )

    returns, lengths, survivals = [], [], []
    losses, noisy_eps, missing_eps, acute_eps = [], [], [], []
    ppo_updates = {
        'policy_loss': [],
        'value_loss': [],
        'entropy': [],
        'clip_fraction': [],
    }

    if checkpoint_eval_seeds is None:
        checkpoint_eval_seeds = [seed]

    best_checkpoint = None
    best_checkpoint_score = (-np.inf, -np.inf)
    best_checkpoint_episode = None
    best_checkpoint_path = None

    obs, info  = env.reset(seed=int(np.random.randint(100_000)))
    ep_return  = 0.0
    ep_steps   = 0
    ep_noisy   = info.get('noisy_episode', False)
    ep_missing = info.get('missing_features') is not None
    ep_acute   = False
    last_update = {'policy_loss': 0.0, 'entropy': 0.0}

    while len(returns) < n_episodes:
        # --- collect one on-policy rollout ---
        b_obs, b_act, b_lp, b_rew, b_val, b_done = [], [], [], [], [], []

        for _ in range(rollout_length):
            action, log_prob, value = agent.act(obs)
            next_obs, r, te, tr, info = env.step(action)
            done = te or tr

            b_obs.append(obs.copy())
            b_act.append(action)
            b_lp.append(log_prob)
            b_rew.append(r)
            b_val.append(value)
            b_done.append(float(done))

            ep_return += r
            ep_steps  += 1
            if info.get('acute_event', False):
                ep_acute = True
            obs = next_obs

            if done:
                returns.append(ep_return)
                lengths.append(ep_steps)
                survivals.append(ep_return > 0)
                losses.append(last_update['policy_loss'])
                noisy_eps.append(ep_noisy)
                missing_eps.append(ep_missing)
                acute_eps.append(ep_acute)

                if verbose and len(returns) % log_every == 0:
                    recent_ret  = np.mean(returns[-log_every:])
                    recent_surv = np.mean(survivals[-log_every:])
                    print(
                        f"[PPO] Ep {len(returns):6d}/{n_episodes} | "
                        f"Return {recent_ret:.3f} | Survival {recent_surv:.2%} | "
                        f"p_loss={last_update['policy_loss']:.4f} "
                        f"entropy={last_update['entropy']:.3f}"
                    )

                should_checkpoint = (
                    checkpoint_every is not None
                    and (
                        len(returns) % checkpoint_every == 0
                        or len(returns) >= n_episodes
                    )
                )

                if should_checkpoint:
                    np_state = np.random.get_state()
                    random_state = random.getstate()
                    torch_state = torch.random.get_rng_state()
                    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

                    eval_result = evaluate_policy_multiseed(
                        agent,
                        eval_seeds=checkpoint_eval_seeds,
                        n_episodes=checkpoint_eval_episodes,
                    )
                    summary = eval_result["summary"]

                    np.random.set_state(np_state)
                    random.setstate(random_state)
                    torch.random.set_rng_state(torch_state)
                    if cuda_state is not None:
                        torch.cuda.set_rng_state_all(cuda_state)

                    score = (summary["survival_rate"], summary["mean_return"])

                    if score > best_checkpoint_score:
                        best_checkpoint_score = score
                        best_checkpoint_episode = len(returns)

                        best_checkpoint = {
                            "algo": "PPO",
                            "episode": best_checkpoint_episode,
                            "score": best_checkpoint_score,
                            "summary": summary,
                            "net": {
                                k: v.detach().cpu().clone()
                                for k, v in agent.net.state_dict().items()
                            },
                            "lr": lr,
                            "gamma": gamma,
                            "gae_lambda": gae_lambda,
                            "clip_eps": clip_eps,
                            "entropy_coef": entropy_coef,
                            "value_coef": value_coef,
                            "rollout_length": rollout_length,
                            "update_epochs": update_epochs,
                            "minibatch_size": minibatch_size,
                            "hidden1": hidden1,
                            "hidden2": hidden2,
                        }

                        if checkpoint_dir is not None:
                            os.makedirs(checkpoint_dir, exist_ok=True)
                            best_checkpoint_path = os.path.join(
                                checkpoint_dir,
                                f"{checkpoint_prefix}_best.pt"
                            )
                            torch.save(best_checkpoint, best_checkpoint_path)

                        if verbose:
                            print(
                                f"[PPO] New best checkpoint at ep {len(returns)}: "
                                f"survival={score[0]:.2%} | return={score[1]:.4f}"
                            )

                if len(returns) >= n_episodes:
                    break

                obs, info = env.reset(seed=int(np.random.randint(100_000)))
                ep_return  = 0.0
                ep_steps   = 0
                ep_noisy   = info.get('noisy_episode', False)
                ep_missing = info.get('missing_features') is not None
                ep_acute   = False

        if len(returns) >= n_episodes:
            break

        # --- bootstrap value and compute GAE ---
        last_value = 0.0 if b_done[-1] else agent.value(obs)

        rewards = np.array(b_rew, dtype=np.float32)
        values  = np.array(b_val + [last_value], dtype=np.float32)
        dones   = np.array(b_done, dtype=np.float32)

        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + gamma * values[t + 1] * nonterminal - values[t]
            gae = delta + gamma * gae_lambda * nonterminal * gae
            advantages[t] = gae
        returns_to_go = advantages + values[:-1]

        last_update = agent.update(
            np.array(b_obs, dtype=np.float32),
            np.array(b_act, dtype=np.int64),
            np.array(b_lp, dtype=np.float32),
            returns_to_go,
            advantages,
        )

        for k in ppo_updates:
            ppo_updates[k].append(last_update[k])

    env.close()

    if best_checkpoint is not None:
        agent.net.load_state_dict(best_checkpoint["net"])
        agent.net.to(agent.device)

    return {
        'agent':       agent,
        'returns':     returns[:n_episodes],
        'lengths':     lengths[:n_episodes],
        'survivals':   survivals[:n_episodes],
        'losses':      losses[:n_episodes],
        'epsilons':    [np.nan] * len(returns[:n_episodes]),
        'noisy_eps':   noisy_eps[:n_episodes],
        'missing_eps': missing_eps[:n_episodes],
        'acute_eps':   acute_eps[:n_episodes],
        'ppo_updates': ppo_updates,
        'best_checkpoint': best_checkpoint,
        'best_checkpoint_episode': best_checkpoint_episode,
        'best_checkpoint_score': best_checkpoint_score,
        'best_checkpoint_path': best_checkpoint_path,
    }


# --- Hyperparameter Tuning (Optuna) ---

def _suggest_dqn_params(trial, double=False):
    """Suggest DQN / Double DQN hyperparameters for an Optuna trial.

    The assignment asks to study buffer_size and target_update_freq explicitly,
    so both are in the search space. Architecture is always symmetric:
    64-64, 128-128 or 256-256.
    """
    arch = trial.suggest_categorical('net_arch', [64, 128, 256])
    return {
        'lr':                   trial.suggest_float('lr', 1e-4, 5e-3, log=True),
        'exploration_fraction': trial.suggest_categorical('exploration_fraction',
                                                           [0.05, 0.10, 0.20]),
        'epsilon_min':          trial.suggest_categorical('epsilon_min',
                                                           [0.01, 0.05, 0.10]),
        'batch_size':           trial.suggest_categorical('batch_size',
                                                           [32, 64, 128]),
        'buffer_size':          trial.suggest_categorical('buffer_size',
                                                           [10_000, 50_000, 100_000]),
        'gradient_steps':       trial.suggest_categorical('gradient_steps',
                                                           [1, 2, 4]),
        'hidden1': arch,
        'hidden2': arch,
    }


def _suggest_ppo_params(trial):
    """Suggest PPO hyperparameters for an Optuna trial."""
    return {
        'lr':             trial.suggest_float('lr', 1e-4, 3e-3, log=True),
        'gae_lambda':     trial.suggest_categorical('gae_lambda',
                                                     [0.90, 0.95, 1.0]),
        'clip_eps':       trial.suggest_categorical('clip_eps',
                                                     [0.1, 0.2, 0.3]),
        'entropy_coef':   trial.suggest_categorical('entropy_coef',
                                                     [0.0, 0.01, 0.05]),
        'value_coef':     trial.suggest_categorical('value_coef', [0.5, 1.0]),
        'rollout_length': trial.suggest_categorical('rollout_length',
                                                     [512, 1024, 2048]),
        'update_epochs':  trial.suggest_categorical('update_epochs', [4, 10]),
        'minibatch_size': trial.suggest_categorical('minibatch_size',
                                                     [64, 128, 256]),
        'hidden1':        trial.suggest_categorical('hidden1', [64, 128, 256]),
        'hidden2':        trial.suggest_categorical('hidden2', [64, 128, 256]),
    }


def run_optuna(
    algo,
    n_trials=10,
    n_episodes_tune=500,
    eval_episodes=200,
    eval_seeds=None,
    metric='combined',
    seed=SEED,
    device=None,
    checkpoint_every=None,
    checkpoint_eval_episodes=1000,
    checkpoint_dir="checkpoints",
    verbose=False,
):
    """
    Tune Double DQN or PPO hyperparameters with Optuna .
    
    Each trial trains one agent, evaluates checkpoints during training, keeps
    the best checkpoint from that trial, and scores the trial using the selected
    checkpoint rather than necessarily the final network weights.

    Returns
    -------
    (dict, optuna.Study, agent, dict)
        best_params, study, best_agent, best_history
    

    """
    if not _HAS_OPTUNA:
        raise ImportError("optuna is not installed. `pip install optuna`.")

    a = algo.lower().replace(' ', '_').replace('-', '_')
    if a not in ('ddqn', 'double_dqn', 'double', 'ppo'):
        raise ValueError(
            f"Unknown algo {algo!r}. Choose 'ddqn', 'double_dqn' or 'ppo'."
        )

    if eval_seeds is None:
        eval_seeds = [seed]

    if checkpoint_every is None:
        trial_checkpoint_every = max(1, n_episodes_tune // 10)
    else:
        trial_checkpoint_every = checkpoint_every

    trial_histories = {}

    def objective(trial):
        # train the agent with the trial's hyperparameters and keep its best validation checkpoint
        if a in ('ddqn', 'double_dqn', 'double'):
            params = _suggest_dqn_params(trial, double=True)
            hist = train_dqn(
                n_episodes=n_episodes_tune,
                double=True,
                seed=seed,
                device=device,
                checkpoint_every=trial_checkpoint_every,
                checkpoint_eval_episodes=checkpoint_eval_episodes,
                checkpoint_eval_seeds=eval_seeds,
                checkpoint_dir=checkpoint_dir,
                checkpoint_prefix=f"optuna_ddqn_trial_{trial.number:03d}",
                **params,
            )

        elif a == 'ppo':
            params = _suggest_ppo_params(trial)
            hist = train_ppo(
                n_episodes=n_episodes_tune,
                seed=seed,
                device=device,
                checkpoint_every=trial_checkpoint_every,
                checkpoint_eval_episodes=checkpoint_eval_episodes,
                checkpoint_eval_seeds=eval_seeds,
                checkpoint_dir=checkpoint_dir,
                checkpoint_prefix=f"optuna_ppo_trial_{trial.number:03d}",
                **params,
            )

        trial_histories[trial.number] = hist

        # evaluate the trained agent with multiple seeds and average the results 
        seed_summaries = []

        for eval_seed in eval_seeds:
            m = evaluate_policy_b(hist['agent'], n_episodes=eval_episodes, seed=eval_seed,)
            seed_summaries.append(m.summary())
        
        survival_rates = np.array([s['survival_rate'] for s in seed_summaries],dtype=float)
        mean_returns = np.array([s['mean_return'] for s in seed_summaries], dtype=float)

        # Average results
        survival_rate_mean = float(survival_rates.mean())
        mean_return_mean = float(mean_returns.mean())

        # Standard deviation across eval seeds (0.0 if only one seed)
        if len(eval_seeds) > 1:
            survival_rate_std = float(survival_rates.std(ddof=1))
            mean_return_std = float(mean_returns.std(ddof=1))
        else:
            survival_rate_std = 0.0
            mean_return_std = 0.0

        # Saving aditional info as trial attributes
        trial.set_user_attr("survival_rate_mean", survival_rate_mean)
        trial.set_user_attr("survival_rate_std", survival_rate_std)
        trial.set_user_attr("mean_return_mean", mean_return_mean)
        trial.set_user_attr("mean_return_std", mean_return_std)
        trial.set_user_attr("eval_seeds", list(eval_seeds))
        trial.set_user_attr("eval_episodes_per_seed", eval_episodes)
        trial.set_user_attr("best_checkpoint_episode", hist.get("best_checkpoint_episode"))
        trial.set_user_attr("best_checkpoint_score", hist.get("best_checkpoint_score"))
        trial.set_user_attr("best_checkpoint_path", hist.get("best_checkpoint_path"))

        # Return the selected metric for optimization
        if metric == 'survival':
            return survival_rate_mean
        elif metric == 'combined':
            return survival_rate_mean, mean_return_mean
        else:
            return mean_return_mean

    # Defining the sampler and study based on the selected metric
    if metric == 'combined':
        study = optuna.create_study(
            directions=['maximize', 'maximize'],
            sampler=optuna.samplers.NSGAIISampler(seed=seed),
        )
    else:
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=seed),
        )

    # Run the optimization
    study.optimize(objective, n_trials=n_trials, show_progress_bar=verbose)

    if metric == 'combined':
        # Multi-objective Optuna returns a Pareto frontier, not one best trial.
        # We choose the clinically strongest candidate: highest survival first,
        # then highest mean return among ties / near-equivalent candidates.
        best_trial = max(
            study.best_trials,
            key=lambda t: (t.values[0], t.values[1])
        )
    else:
        best_trial = study.best_trial
    
    best_training_hist = trial_histories.get(best_trial.number)

    # Optuna returns parameter names as registered with suggest_*.
    # Convert 'net_arch' (used internally) to 'hidden1'/'hidden2' so the
    # returned dict plugs directly into train_dqn / train_ppo.
    best = dict(best_trial.params)
    if 'net_arch' in best:
        arch = best.pop('net_arch')
        best['hidden1'] = arch
        best['hidden2'] = arch
    
    # Extract the best checkpoint path from the trial's user attributes, if it exists
    best_checkpoint_path = best_trial.user_attrs.get("best_checkpoint_path")

    # Initialize variables to hold the best agent and its history, which we will load from the best checkpoint if it exists
    best_agent = None
    best_hist = None

    # Load the best checkpoint's weights into a new agent instance so that we can return the best agent found during tuning
    # along with its checkpoint info in best_hist.
    if best_checkpoint_path is not None:
        checkpoint = torch.load(best_checkpoint_path,
            map_location=torch.device(
                device if device is not None else get_default_device()))

        if a in ('ddqn', 'double_dqn', 'double'):
            best_agent = DQNAgent(
                lr=best["lr"],
                gamma=GAMMA,
                epsilon_min=best.get("epsilon_min", 0.05),
                buffer_size=best["buffer_size"],
                batch_size=best["batch_size"],
                hidden1=best["hidden1"],
                hidden2=best["hidden2"],
                double=True,
                device=device,
            )

            best_agent.online_net.load_state_dict(checkpoint["online_net"])
            best_agent.target_net.load_state_dict(checkpoint["target_net"])
            best_agent.epsilon = checkpoint.get("epsilon", best_agent.epsilon)

        elif a == 'ppo':
            best_agent = PPOAgent(
                lr=best["lr"],
                gamma=GAMMA,
                gae_lambda=best["gae_lambda"],
                clip_eps=best["clip_eps"],
                entropy_coef=best["entropy_coef"],
                value_coef=best["value_coef"],
                rollout_length=best["rollout_length"],
                update_epochs=best["update_epochs"],
                minibatch_size=best["minibatch_size"],
                hidden1=best["hidden1"],
                hidden2=best["hidden2"],
                device=device,
            )

            best_agent.net.load_state_dict(checkpoint["net"])

        # We construct the best_hist dict to include all the checkpoint info and training history for the best trial, 
        # so that we can use it later for analysis and plotting
        best_hist = {
            "agent": best_agent,
            "best_checkpoint": checkpoint,
            "best_checkpoint_episode": checkpoint.get("episode"),
            "best_checkpoint_score": checkpoint.get("score"),
            "best_checkpoint_path": best_checkpoint_path,
            "best_trial_number": best_trial.number,
            "best_trial_values": best_trial.values,
        }

        if best_training_hist is not None:
            best_hist.update({
                "returns": best_training_hist["returns"],
                "lengths": best_training_hist["lengths"],
                "survivals": best_training_hist["survivals"],
                "losses": best_training_hist["losses"],
                "epsilons": best_training_hist["epsilons"],
                "noisy_eps": best_training_hist["noisy_eps"],
                "missing_eps": best_training_hist["missing_eps"],
                "acute_eps": best_training_hist["acute_eps"],
            })

            if "ppo_updates" in best_training_hist:
                best_hist["ppo_updates"] = best_training_hist["ppo_updates"]

    # Fallback: if no checkpoint file was available, use the in-memory best trial history.
    if best_hist is None and best_training_hist is not None:
        best_agent = best_training_hist["agent"]

        best_hist = {
            "agent": best_agent,
            "best_checkpoint": best_training_hist.get("best_checkpoint"),
            "best_checkpoint_episode": best_training_hist.get("best_checkpoint_episode"),
            "best_checkpoint_score": best_training_hist.get("best_checkpoint_score"),
            "best_checkpoint_path": best_training_hist.get("best_checkpoint_path"),
            "best_trial_number": best_trial.number,
            "best_trial_values": best_trial.values,
            "returns": best_training_hist["returns"],
            "lengths": best_training_hist["lengths"],
            "survivals": best_training_hist["survivals"],
            "losses": best_training_hist["losses"],
            "epsilons": best_training_hist["epsilons"],
            "noisy_eps": best_training_hist["noisy_eps"],
            "missing_eps": best_training_hist["missing_eps"],
            "acute_eps": best_training_hist["acute_eps"],
        }

        if "ppo_updates" in best_training_hist:
            best_hist["ppo_updates"] = best_training_hist["ppo_updates"]

    return best, study, best_agent, best_hist

#---------------------------------------------------------------------------------------------------------------------------------------------------
                                                            # Note on Optuna trials:
# In Optuna, a `trial` represents one hyperparameter attempt. We do not create trial objects manually. 
# When study.optimize(...) is called, Optuna creates one Trial object per attempt and passes it into the objective  and _suggest_..._params functions. 
# The trial object is then used to sample hyperparameters and to store extra evaluation statistics such as multi-seed means and standard deviations.
#---------------------------------------------------------------------------------------------------------------------------------------------------

# --- Final Evaluation ---
def evaluate_policy_multiseed(agent, eval_seeds, n_episodes):
    """ Evaluate the given agent across multiple seeds and return overall and
    stratified summaries.

    Parameters
    ----------
    agent : DQNAgent or PPOAgent
        The trained agent to evaluate.
    eval_seeds : list[int]
        List of random seeds to use for evaluation. 
    n_episodes : int
        Number of episodes to run for each seed.

    Returns
    -------
    dict
        'summary': overall mean/std across seeds.
        'per_seed': per-seed overall metrics.
        'stratified': mean/std metrics by clinical failure mode.
        'stratified_per_seed': per-seed stratified metrics.
        'metrics_by_seed': raw EvalMetricsB objects for further analysis.
    """

    rows = []
    stratified_rows = []
    metrics_by_seed = {}

    for eval_seed in eval_seeds:
        m = evaluate_policy_b(agent, n_episodes=n_episodes,seed=eval_seed,)
        s = m.summary()

        # Save results from this seed
        rows.append({
            "seed": eval_seed,
            "mean_return": s["mean_return"],
            "survival_rate": s["survival_rate"],
            "mean_length": s["mean_length"],
        })
        metrics_by_seed[eval_seed] = m

        # Stratified results by clinical failure mode
        # Create boolean masks for each failure mode across all episodes in this seed's evaluation
        returns_arr = np.array(m.episode_returns)
        noisy_mask = np.array(m.noisy_flags)
        missing_mask = np.array(m.missing_flags)
        acute_mask = np.array(m.acute_flags)

        strata = [("Noisy", noisy_mask), ("Clean", ~noisy_mask),
                  ("Missing obs", missing_mask), ("Complete obs", ~missing_mask),
                ("Acute event", acute_mask),("No acute", ~acute_mask)]

        for label, mask in strata:
            if mask.sum() == 0:
                continue

            selected_returns = returns_arr[mask]

            stratified_rows.append({
                "seed": eval_seed,
                "group": label,
                "n_episodes": int(mask.sum()),
                "mean_return": float(selected_returns.mean()),
                "survival_rate": float((selected_returns > 0).mean()),
            })


    df = pd.DataFrame(rows)
    stratified_per_seed_df = pd.DataFrame(stratified_rows)

    # Aggregate results across seeds to get overall mean and std for each metric
    summary = {
        "mean_return": float(df["mean_return"].mean()),
        "mean_return_std": float(df["mean_return"].std(ddof=1)),
        "survival_rate": float(df["survival_rate"].mean()),
        "survival_rate_std": float(df["survival_rate"].std(ddof=1)),
        "mean_length": float(df["mean_length"].mean()),
        "mean_length_std": float(df["mean_length"].std(ddof=1)),
    }

    # Aggregate stratified results across seeds to get mean and std for each group
    stratified_df = (stratified_per_seed_df.groupby("group")
        .agg(
            mean_return=("mean_return", "mean"),
            mean_return_std=("mean_return", "std"),
            survival_rate=("survival_rate", "mean"),
            survival_rate_std=("survival_rate", "std"),
        )
        .reset_index()
    )

    return {"summary": summary, "per_seed": df, "stratified": stratified_df, "stratified_per_seed": stratified_per_seed_df, "metrics_by_seed": metrics_by_seed}

# --- Plotting utilities ---

def moving_average(x, window=100):
    """
    Causal moving average that preserves the input length.

    The first (window−1) values use a shorter prefix window so the output
    is always the same length as the input. This allows all learning curves
    to be plotted on the same episode x-axis without alignment offsets.

    Parameters
    ----------
    x      : list[float] or np.ndarray
    window : int  – averaging window size

    Returns
    -------
    np.ndarray  same length as x
    """
    v = np.asarray(x, dtype=float)
    if v.size == 0:
        return v
    w = int(min(window, v.size))
    cumsum = np.cumsum(np.insert(v, 0, 0.0))
    full   = (cumsum[w:] - cumsum[:-w]) / w
    head   = np.array([v[:i + 1].mean() for i in range(w - 1)])
    return np.concatenate([head, full])


def plot_return_curves(
    histories,
    window=500,
    title='Learning Curves — Episode Return',
    baselines=None,
    max_episodes=None,
    figsize=(11, 5),
    savepath=None,
):
    """
    One figure comparing smoothed return learning curves for all algorithms.

    Parameters
    ----------
    histories : dict[str, dict]
        Maps label -> training history dict from train_dqn / train_ppo.
    window : int
        Moving-average smoothing window.
    baselines : dict[str, float] or None
        Optional horizontal reference lines {label: value}, e.g. random baseline.
    max_episodes : int or None
        Fix x-axis upper bound for a consistent comparison.
    savepath : str or None
        If given, save the figure to this path.
    """
    plt.figure(figsize=figsize)
    for label, h in histories.items():
        y = moving_average(h['returns'], window)
        plt.plot(np.arange(len(y)), y, label=label, linewidth=2)
    if baselines:
        for blabel, bval in baselines.items():
            plt.axhline(bval, linestyle='--', linewidth=1.2, label=blabel, color='gray')
    plt.xlabel('Training episode')
    plt.ylabel(f'Return (MA-{window})')
    plt.title(title)
    if max_episodes is not None:
        plt.xlim(0, max_episodes)
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=120, bbox_inches='tight')
    plt.show()


def plot_survival_curves(
    histories,
    window=500,
    title='Learning Curves — Survival Rate',
    baselines=None,
    max_episodes=None,
    figsize=(11, 5),
    savepath=None,
):
    """
    One figure comparing smoothed survival-rate learning curves for all algorithms.

    Parameters identical to plot_return_curves but the y-axis shows the
    fraction of episodes in which the patient survived (return > 0).
    """
    plt.figure(figsize=figsize)
    for label, h in histories.items():
        y = moving_average([float(s) for s in h['survivals']], window)
        plt.plot(np.arange(len(y)), y, label=label, linewidth=2)
    if baselines:
        for blabel, bval in baselines.items():
            plt.axhline(bval, linestyle='--', linewidth=1.2, label=blabel, color='gray')
    plt.xlabel('Training episode')
    plt.ylabel(f'Survival rate (MA-{window})')
    plt.ylim(0, 1)
    plt.title(title)
    if max_episodes is not None:
        plt.xlim(0, max_episodes)
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=120, bbox_inches='tight')
    plt.show()


# --- Convergence utilities ---

def dqn_convergence_episode(returns, window=100, patience=5, threshold=0.02):
    """
    Estimate the episode at which training converged.

    Convergence is declared when the 'valid'-mode moving average changes by
    less than threshold for patience consecutive steps.

    Returns
    -------
    int  - estimated convergence episode; len(returns) if never detected.
    """
    ma = np.convolve(
        np.array(returns, dtype=float), np.ones(window) / window, mode='valid'
    )
    stagnant = 0
    for i in range(1, len(ma)):
        if abs(ma[i] - ma[i - 1]) < threshold:
            stagnant += 1
        else:
            stagnant = 0
        if stagnant >= patience:
            return i + window - 1
    return len(returns)