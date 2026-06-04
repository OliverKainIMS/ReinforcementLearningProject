"""
rl_configB_functions_v2.py — enhanced learning code for Config B
================================================================
This module is an ADD-ON to rl_configB_functions.py. The ENVIRONMENT is left
completely untouched: it still uses the required make_clinical_env() from
envs.wrappers (same rewards, same dynamics, same failure modes). All changes
live in the *agent* and the *training loops*.

What changed vs the original
----------------------------
1. In-model observation history (frame stacking).
   The clinical wrappers make ~30% of episodes partially observable (noisy or
   missing features). A memoryless network cannot cope with that. Here the
   *agent* keeps the last `frame_stack` observations and feeds their
   concatenation to the network. The environment never sees this — the history
   is built inside the training/eval loop with a small FrameStacker helper.

2. n-step returns for DQN.
   With a sparse terminal survival reward, 1-step bootstrapping propagates the
   signal very slowly. NStepReplayBuffer accumulates n transitions so the
   survival reward reaches earlier states faster.

3. Decoupled update schedule.
   `train_freq` (update every k environment steps) + larger default
   `batch_size` reduce over-fitting to the heavy reward noise. The default
   `target_update_freq` is also less aggressive.

4. Best-checkpoint selection (the Lab 6 lesson).
   DQN/PPO often end training on worse weights than they had mid-run. Both
   training loops now evaluate the policy periodically on a held-out
   validation seed and KEEP THE BEST checkpoint, returning that agent instead
   of the final one. Optuna therefore scores each trial at its best checkpoint.

Public names are re-exported with the SAME names as the original module, so a
notebook only needs to change its import line to switch to v2.
"""

import copy
import random
from collections import deque

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Unchanged building blocks reused from the original module.
from rl_configB_functions import (
    SEED, N_OBS, N_ACTIONS, GAMMA,
    get_default_device,
    QNetwork, ActorCritic, PPOAgent,
    EvalMetricsB,
    moving_average, plot_return_curves, plot_survival_curves,
    dqn_convergence_episode,
    optuna, _HAS_OPTUNA,
)

# The REQUIRED environment, unchanged. Frame stacking is applied loop-side.
from envs.wrappers import make_clinical_env


# ---------------------------------------------------------------------------
#  In-model observation history (frame stacking)
# ---------------------------------------------------------------------------
class FrameStacker:
    """
    Maintains the last `frame_stack` observations and returns their flat
    concatenation. Lives inside the agent's training/eval loop — it does NOT
    modify the environment.

    frame_stack == 1 is a no-op (returns the single observation), so the v2
    code reduces exactly to the original single-frame behaviour.
    """

    def __init__(self, frame_stack, base_dim):
        self.k = max(1, int(frame_stack))
        self.base_dim = int(base_dim)
        self.frames = deque(maxlen=self.k)

    def reset(self, obs):
        self.frames.clear()
        f = np.asarray(obs, dtype=np.float32)
        for _ in range(self.k):
            self.frames.append(f)
        return self._get()

    def append(self, obs):
        self.frames.append(np.asarray(obs, dtype=np.float32))
        return self._get()

    def _get(self):
        if self.k <= 1:
            return np.asarray(self.frames[-1], dtype=np.float32)
        return np.concatenate(list(self.frames)).astype(np.float32)


def make_stacker_for(agent):
    """Build a FrameStacker matching an agent's frame_stack / base_obs_dim.

    Convenience for the notebook analysis cells (ensemble, trajectories) so
    they can feed correctly stacked observations to a trained agent.
    """
    fs = getattr(agent, 'frame_stack', 1)
    base = getattr(agent, 'base_obs_dim', N_OBS)
    return FrameStacker(fs, base)


# ---------------------------------------------------------------------------
#  n-step replay buffer
# ---------------------------------------------------------------------------
class NStepReplayBuffer:
    """
    Uniform replay buffer that emits n-step transitions.

    Each stored transition is (obs, action, R, next_obs, done, discount) where
        R        = sum_{i=0}^{m-1} gamma^i r_{t+i}     (m <= n_step, truncated
                   at episode end)
        next_obs = observation m steps ahead
        discount = gamma^m  (the discount applied to the bootstrap Q value)

    With n_step == 1 this is identical to the original 1-step ReplayBuffer.
    """

    def __init__(self, capacity=50_000, n_step=1, gamma=1.0):
        self.buffer = deque(maxlen=capacity)
        self.n_step = max(1, int(n_step))
        self.gamma = float(gamma)
        self.n_queue = deque(maxlen=self.n_step)

    def _make(self, queue):
        obs, action = queue[0][0], queue[0][1]
        R, boot, next_obs, done = 0.0, 1.0, queue[0][3], 0.0
        for (_o, _a, r, no, d) in queue:
            R += boot * r
            boot *= self.gamma
            next_obs = no
            if d:
                done = 1.0
                break
        return (obs, action, R, next_obs, done, boot)

    def push(self, obs, action, reward, next_obs, done):
        self.n_queue.append((obs, action, reward, next_obs, float(done)))
        if not done:
            if len(self.n_queue) >= self.n_step:
                self.buffer.append(self._make(self.n_queue))
        else:
            # Episode ended: flush an n-step transition for every start index.
            while self.n_queue:
                self.buffer.append(self._make(self.n_queue))
                self.n_queue.popleft()

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs, actions, R, next_obs, dones, disc = zip(*batch)
        return (
            np.array(obs,      dtype=np.float32),
            np.array(actions,  dtype=np.int64),
            np.array(R,        dtype=np.float32),
            np.array(next_obs, dtype=np.float32),
            np.array(dones,    dtype=np.float32),
            np.array(disc,     dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# Re-exported name (the notebook imports `ReplayBuffer`).
ReplayBuffer = NStepReplayBuffer


# ---------------------------------------------------------------------------
#  DQN / Double DQN agent with n-step targets
# ---------------------------------------------------------------------------
class DQNAgent:
    """
    DQN / Double DQN agent. Identical to the original except:
      * the network input dimension is base_obs_dim * frame_stack (the agent
        consumes a stacked observation built by the training/eval loop);
      * it uses an n-step replay buffer, so the bootstrap target is
            y = R + (gamma^m) * Q_target(s_{t+m}) * (1 - done).
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
        batch_size=128,
        target_update_freq=250,
        n_step=1,
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
        self.target_update_freq = target_update_freq
        self.buffer_size        = buffer_size
        self.n_step             = n_step
        self.double             = double
        self.device             = torch.device(device if device is not None else get_default_device())

        self.online_net = QNetwork(obs_dim, n_actions, hidden1, hidden2).to(self.device)
        self.target_net = QNetwork(obs_dim, n_actions, hidden1, hidden2).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer     = optim.Adam(self.online_net.parameters(), lr=lr)
        self.replay_buffer = NStepReplayBuffer(buffer_size, n_step=n_step, gamma=gamma)
        self.steps_done    = 0

        # Set by the training loop so eval/analysis code can rebuild the stack.
        self.frame_stack   = 1
        self.base_obs_dim  = obs_dim

    def select_action(self, obs, greedy=False):
        """Epsilon-greedy action selection on a (already stacked) observation."""
        if not greedy and np.random.rand() < self.epsilon:
            return int(np.random.randint(self.n_actions))
        obs_t = torch.tensor(
            np.array(obs, dtype=np.float32), device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            return int(self.online_net(obs_t).argmax(dim=1).item())

    def update(self):
        """One n-step gradient step. Returns 0.0 if the buffer is too small."""
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        obs, actions, R, next_obs, dones, disc = self.replay_buffer.sample(self.batch_size)

        obs_t      = torch.tensor(obs,      device=self.device)
        actions_t  = torch.tensor(actions,  device=self.device).unsqueeze(1)
        R_t        = torch.tensor(R,        device=self.device)
        next_obs_t = torch.tensor(next_obs, device=self.device)
        dones_t    = torch.tensor(dones,    device=self.device)
        disc_t     = torch.tensor(disc,     device=self.device)

        q_current = self.online_net(obs_t).gather(1, actions_t).squeeze(1)

        with torch.no_grad():
            if self.double:
                next_actions = self.online_net(next_obs_t).argmax(dim=1, keepdim=True)
                q_next = self.target_net(next_obs_t).gather(1, next_actions).squeeze(1)
            else:
                q_next = self.target_net(next_obs_t).max(dim=1).values
            q_target = R_t + disc_t * q_next * (1.0 - dones_t)

        loss = nn.functional.smooth_l1_loss(q_current, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.steps_done += 1
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return float(loss.item())


# ---------------------------------------------------------------------------
#  Checkpoint helper: quick greedy validation
# ---------------------------------------------------------------------------
def _clone_state(net):
    return {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}


def _quick_eval(agent, n_episodes, seed, frame_stack):
    """Greedy validation on held-out seeds. Returns (mean_return, survival_rate)."""
    env = make_clinical_env()
    base_dim = env.observation_space.shape[0]
    stacker = FrameStacker(frame_stack, base_dim)
    np.random.seed(seed)
    rets, survs = [], []
    for _ in range(n_episodes):
        obs, _info = env.reset(seed=int(np.random.randint(100_000)))
        s = stacker.reset(obs)
        done, total_r = False, 0.0
        while not done:
            a = agent.select_action(s, greedy=True)
            obs, r, te, tr, _info = env.step(a)
            s = stacker.append(obs)
            total_r += r
            done = te or tr
        rets.append(total_r)
        survs.append(total_r > 0)
    env.close()
    return float(np.mean(rets)), float(np.mean(survs))


# ---------------------------------------------------------------------------
#  Evaluation (uses the agent's own frame_stack)
# ---------------------------------------------------------------------------
def evaluate_policy_b(agent, n_episodes=1000, seed=SEED, **env_kwargs):
    """Evaluate a trained agent on make_clinical_env(); frame-stacks per episode."""
    env = make_clinical_env(**env_kwargs)
    base_dim = env.observation_space.shape[0]
    stacker = FrameStacker(getattr(agent, 'frame_stack', 1), base_dim)
    metrics = EvalMetricsB()
    np.random.seed(seed)

    for _ in range(n_episodes):
        obs, info = env.reset(seed=int(np.random.randint(100_000)))
        s = stacker.reset(obs)
        done       = False
        total_r    = 0.0
        steps      = 0
        ep_noisy   = info.get('noisy_episode', False)
        ep_missing = info.get('missing_features') is not None
        ep_acute   = False

        while not done:
            action = agent.select_action(s, greedy=True)
            obs, r, te, tr, info = env.step(action)
            s = stacker.append(obs)
            total_r += r
            steps   += 1
            done     = te or tr
            if info.get('acute_event', False):
                ep_acute = True

        metrics.add(total_r, steps, noisy=ep_noisy, missing=ep_missing, acute=ep_acute)

    env.close()
    return metrics


def evaluate_policy_multiseed(agent, eval_seeds, n_episodes):
    """Multi-seed evaluation with overall and stratified summaries (v2 env-free)."""
    rows = []
    stratified_rows = []
    metrics_by_seed = {}

    for eval_seed in eval_seeds:
        m = evaluate_policy_b(agent, n_episodes=n_episodes, seed=eval_seed)
        s = m.summary()

        rows.append({
            "seed": eval_seed,
            "mean_return": s["mean_return"],
            "survival_rate": s["survival_rate"],
            "mean_length": s["mean_length"],
        })
        metrics_by_seed[eval_seed] = m

        returns_arr  = np.array(m.episode_returns)
        noisy_mask   = np.array(m.noisy_flags)
        missing_mask = np.array(m.missing_flags)
        acute_mask   = np.array(m.acute_flags)

        strata = [("Noisy", noisy_mask), ("Clean", ~noisy_mask),
                  ("Missing obs", missing_mask), ("Complete obs", ~missing_mask),
                  ("Acute event", acute_mask), ("No acute", ~acute_mask)]

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

    summary = {
        "mean_return": float(df["mean_return"].mean()),
        "mean_return_std": float(df["mean_return"].std(ddof=1)),
        "survival_rate": float(df["survival_rate"].mean()),
        "survival_rate_std": float(df["survival_rate"].std(ddof=1)),
        "mean_length": float(df["mean_length"].mean()),
        "mean_length_std": float(df["mean_length"].std(ddof=1)),
    }

    stratified_df = (stratified_per_seed_df.groupby("group")
        .agg(
            mean_return=("mean_return", "mean"),
            mean_return_std=("mean_return", "std"),
            survival_rate=("survival_rate", "mean"),
            survival_rate_std=("survival_rate", "std"),
        )
        .reset_index()
    )

    return {"summary": summary, "per_seed": df, "stratified": stratified_df,
            "stratified_per_seed": stratified_per_seed_df, "metrics_by_seed": metrics_by_seed}


# ---------------------------------------------------------------------------
#  DQN training loop (frame stacking + n-step + train_freq + checkpointing)
# ---------------------------------------------------------------------------
def train_dqn(
    n_episodes=50_000,
    double=False,
    lr=1e-3,
    gamma=GAMMA,
    epsilon_start=1.0,
    epsilon_min=0.05,
    exploration_fraction=0.05,
    buffer_size=50_000,
    batch_size=128,
    target_update_freq=250,
    gradient_steps=1,
    train_freq=4,
    n_step=3,
    frame_stack=4,
    hidden1=128,
    hidden2=128,
    learning_starts=1000,
    seed=SEED,
    device=None,
    verbose=False,
    log_every=1000,
    select_best=True,
    ckpt_eval_every=None,
    ckpt_eval_episodes=300,
    ckpt_val_seed=987_654,
    ckpt_metric='survival',
):
    """
    Train DQN / Double DQN with in-model frame stacking, n-step returns, a
    decoupled update schedule and best-checkpoint selection.

    The returned agent is the BEST checkpoint found on the held-out validation
    seed (not necessarily the final weights). Extra history keys 'ckpt_episodes',
    'ckpt_scores', 'best_ep', 'best_score' describe the validation curve.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    env       = make_clinical_env()
    algo_name = 'Double DQN' if double else 'DQN'
    base_dim  = env.observation_space.shape[0]
    obs_dim   = base_dim * max(1, int(frame_stack))

    agent = DQNAgent(
        obs_dim=obs_dim, n_actions=N_ACTIONS,
        lr=lr, gamma=gamma,
        epsilon_start=epsilon_start, epsilon_min=epsilon_min,
        buffer_size=buffer_size, batch_size=batch_size,
        target_update_freq=target_update_freq, n_step=n_step,
        hidden1=hidden1, hidden2=hidden2,
        double=double, device=device,
    )
    agent.frame_stack  = max(1, int(frame_stack))
    agent.base_obs_dim = base_dim

    stacker        = FrameStacker(frame_stack, base_dim)
    decay_episodes = max(1, int(exploration_fraction * n_episodes))
    if ckpt_eval_every is None:
        ckpt_eval_every = max(1, n_episodes // 20)

    returns, lengths, survivals = [], [], []
    losses, epsilons            = [], []
    noisy_eps, missing_eps, acute_eps = [], [], []
    ckpt_episodes, ckpt_scores  = [], []
    best_score, best_state, best_ep = -np.inf, None, None
    total_steps = 0

    for ep in range(n_episodes):
        agent.epsilon = max(
            epsilon_min,
            epsilon_start - (epsilon_start - epsilon_min) * ep / decay_episodes,
        )

        obs, info  = env.reset(seed=int(np.random.randint(100_000)))
        s          = stacker.reset(obs)
        done       = False
        total_r    = 0.0
        steps      = 0
        ep_losses  = []
        ep_noisy   = info.get('noisy_episode', False)
        ep_missing = info.get('missing_features') is not None
        ep_acute   = False

        while not done:
            action = agent.select_action(s)
            next_obs, r, te, tr, info = env.step(action)
            done   = te or tr
            s_next = stacker.append(next_obs)
            agent.replay_buffer.push(s, action, r, s_next, float(done))
            total_steps += 1

            if total_steps >= learning_starts and total_steps % train_freq == 0:
                for _ in range(gradient_steps):
                    loss = agent.update()
                    if loss > 0.0:
                        ep_losses.append(loss)

            s        = s_next
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

        # --- best-checkpoint selection on a held-out validation seed ---
        if select_best and (ep + 1) % ckpt_eval_every == 0:
            v_ret, v_surv = _quick_eval(agent, ckpt_eval_episodes, ckpt_val_seed, agent.frame_stack)
            score = v_surv if ckpt_metric == 'survival' else v_ret
            ckpt_episodes.append(ep + 1)
            ckpt_scores.append(score)
            if score > best_score:
                best_score = score
                best_state = _clone_state(agent.online_net)
                best_ep    = ep + 1
            if verbose:
                print(f"[{algo_name}] ckpt ep {ep+1:6d} | "
                      f"val_surv={v_surv:.2%} val_ret={v_ret:.3f} | "
                      f"best={best_score:.4f} @ ep {best_ep}")

        if verbose and (ep + 1) % log_every == 0:
            recent_ret  = np.mean(returns[-log_every:])
            recent_surv = np.mean(survivals[-log_every:])
            print(
                f"[{algo_name}] Ep {ep+1:6d}/{n_episodes} | "
                f"Return {recent_ret:.3f} | Survival {recent_surv:.2%} | "
                f"ε={agent.epsilon:.4f} | buffer={len(agent.replay_buffer):6d}"
            )

    env.close()

    # Restore the best checkpoint (the Lab 6 lesson: last weights are rarely best).
    if select_best and best_state is not None:
        agent.online_net.load_state_dict(best_state)
        agent.target_net.load_state_dict(best_state)

    return {
        'agent':         agent,
        'returns':       returns,
        'lengths':       lengths,
        'survivals':     survivals,
        'losses':        losses,
        'epsilons':      epsilons,
        'noisy_eps':     noisy_eps,
        'missing_eps':   missing_eps,
        'acute_eps':     acute_eps,
        'ckpt_episodes': ckpt_episodes,
        'ckpt_scores':   ckpt_scores,
        'best_ep':       best_ep,
        'best_score':    best_score,
        'frame_stack':   agent.frame_stack,
        'n_step':        n_step,
    }


# ---------------------------------------------------------------------------
#  PPO training loop (frame stacking + checkpointing)
# ---------------------------------------------------------------------------
def train_ppo(
    n_episodes=50_000,
    lr=3e-4,
    gamma=GAMMA,
    gae_lambda=0.95,
    clip_eps=0.2,
    entropy_coef=0.01,
    value_coef=0.5,
    rollout_length=1024,
    update_epochs=10,
    minibatch_size=128,
    frame_stack=4,
    hidden1=128,
    hidden2=128,
    seed=SEED,
    device=None,
    verbose=False,
    log_every=1000,
    select_best=True,
    ckpt_eval_every=None,
    ckpt_eval_episodes=300,
    ckpt_val_seed=987_654,
    ckpt_metric='survival',
):
    """
    Train PPO with in-model frame stacking and best-checkpoint selection.
    Returns the best-checkpoint agent (not the final one).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    env      = make_clinical_env()
    base_dim = env.observation_space.shape[0]
    obs_dim  = base_dim * max(1, int(frame_stack))

    agent = PPOAgent(
        obs_dim=obs_dim, n_actions=N_ACTIONS,
        lr=lr, gamma=gamma, gae_lambda=gae_lambda,
        clip_eps=clip_eps, entropy_coef=entropy_coef, value_coef=value_coef,
        rollout_length=rollout_length, update_epochs=update_epochs,
        minibatch_size=minibatch_size,
        hidden1=hidden1, hidden2=hidden2, device=device,
    )
    agent.frame_stack  = max(1, int(frame_stack))
    agent.base_obs_dim = base_dim

    stacker = FrameStacker(frame_stack, base_dim)
    if ckpt_eval_every is None:
        ckpt_eval_every = max(1, n_episodes // 20)

    returns, lengths, survivals = [], [], []
    losses, noisy_eps, missing_eps, acute_eps = [], [], [], []
    ppo_updates = {'policy_loss': [], 'value_loss': [], 'entropy': [], 'clip_fraction': []}
    ckpt_episodes, ckpt_scores = [], []
    best_score, best_state, best_ep = -np.inf, None, None

    obs, info  = env.reset(seed=int(np.random.randint(100_000)))
    s          = stacker.reset(obs)
    ep_return  = 0.0
    ep_steps   = 0
    ep_noisy   = info.get('noisy_episode', False)
    ep_missing = info.get('missing_features') is not None
    ep_acute   = False
    last_update = {'policy_loss': 0.0, 'entropy': 0.0}

    def _maybe_checkpoint():
        nonlocal best_score, best_state, best_ep
        n_done = len(returns)
        if select_best and n_done > 0 and n_done % ckpt_eval_every == 0:
            v_ret, v_surv = _quick_eval(agent, ckpt_eval_episodes, ckpt_val_seed, agent.frame_stack)
            score = v_surv if ckpt_metric == 'survival' else v_ret
            ckpt_episodes.append(n_done)
            ckpt_scores.append(score)
            if score > best_score:
                best_score = score
                best_state = _clone_state(agent.net)
                best_ep    = n_done
            if verbose:
                print(f"[PPO] ckpt ep {n_done:6d} | val_surv={v_surv:.2%} "
                      f"val_ret={v_ret:.3f} | best={best_score:.4f} @ ep {best_ep}")

    while len(returns) < n_episodes:
        b_obs, b_act, b_lp, b_rew, b_val, b_done = [], [], [], [], [], []

        for _ in range(rollout_length):
            action, log_prob, value = agent.act(s)
            next_obs, r, te, tr, info = env.step(action)
            done = te or tr

            b_obs.append(s.copy())
            b_act.append(action)
            b_lp.append(log_prob)
            b_rew.append(r)
            b_val.append(value)
            b_done.append(float(done))

            ep_return += r
            ep_steps  += 1
            if info.get('acute_event', False):
                ep_acute = True
            s = stacker.append(next_obs)

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

                _maybe_checkpoint()

                if len(returns) >= n_episodes:
                    break

                obs, info  = env.reset(seed=int(np.random.randint(100_000)))
                s          = stacker.reset(obs)
                ep_return  = 0.0
                ep_steps   = 0
                ep_noisy   = info.get('noisy_episode', False)
                ep_missing = info.get('missing_features') is not None
                ep_acute   = False

        # --- GAE advantage computation ---
        last_value = 0.0 if b_done[-1] == 1.0 else agent.value(s)
        adv = np.zeros(len(b_rew), dtype=np.float32)
        gae = 0.0
        for t in reversed(range(len(b_rew))):
            next_val    = last_value if t == len(b_rew) - 1 else b_val[t + 1]
            nonterminal = 1.0 - b_done[t]
            delta       = b_rew[t] + gamma * next_val * nonterminal - b_val[t]
            gae         = delta + gamma * gae_lambda * nonterminal * gae
            adv[t]      = gae
        b_returns = adv + np.array(b_val, dtype=np.float32)

        metrics = agent.update(
            np.array(b_obs, dtype=np.float32),
            np.array(b_act, dtype=np.int64),
            np.array(b_lp,  dtype=np.float32),
            b_returns, adv,
        )
        last_update = metrics
        for k in ppo_updates:
            ppo_updates[k].append(metrics[k])

    env.close()

    if select_best and best_state is not None:
        agent.net.load_state_dict(best_state)

    return {
        'agent':         agent,
        'returns':       returns[:n_episodes],
        'lengths':       lengths[:n_episodes],
        'survivals':     survivals[:n_episodes],
        'losses':        losses[:n_episodes],
        'epsilons':      [float('nan')] * min(len(returns), n_episodes),
        'noisy_eps':     noisy_eps[:n_episodes],
        'missing_eps':   missing_eps[:n_episodes],
        'acute_eps':     acute_eps[:n_episodes],
        'ppo_updates':   ppo_updates,
        'ckpt_episodes': ckpt_episodes,
        'ckpt_scores':   ckpt_scores,
        'best_ep':       best_ep,
        'best_score':    best_score,
        'frame_stack':   agent.frame_stack,
    }


# ---------------------------------------------------------------------------
#  Hyperparameter tuning (Optuna) — scores each trial at its best checkpoint
# ---------------------------------------------------------------------------
def _suggest_dqn_params(trial, double=False):
    """DQN / Double DQN search space (adds n_step, train_freq, frame_stack)."""
    arch = trial.suggest_categorical('net_arch', [64, 128, 256])
    return {
        'lr':                   trial.suggest_float('lr', 1e-4, 5e-3, log=True),
        'exploration_fraction': trial.suggest_categorical('exploration_fraction',
                                                           [0.05, 0.10, 0.20]),
        'epsilon_min':          trial.suggest_categorical('epsilon_min',
                                                           [0.01, 0.05, 0.10]),
        'batch_size':           trial.suggest_categorical('batch_size',
                                                           [64, 128, 256]),
        'buffer_size':          trial.suggest_categorical('buffer_size',
                                                           [10_000, 50_000, 100_000]),
        'target_update_freq':   trial.suggest_categorical('target_update_freq',
                                                           [100, 250, 500]),
        'gradient_steps':       trial.suggest_categorical('gradient_steps', [1, 2]),
        'train_freq':           trial.suggest_categorical('train_freq', [1, 4, 8]),
        'n_step':               trial.suggest_categorical('n_step', [1, 3, 5]),
        'frame_stack':          trial.suggest_categorical('frame_stack', [1, 2, 4]),
        'hidden1': arch,
        'hidden2': arch,
    }


def _suggest_ppo_params(trial):
    """PPO search space (adds frame_stack)."""
    return {
        'lr':             trial.suggest_float('lr', 1e-4, 3e-3, log=True),
        'gae_lambda':     trial.suggest_categorical('gae_lambda', [0.90, 0.95, 1.0]),
        'clip_eps':       trial.suggest_categorical('clip_eps', [0.1, 0.2, 0.3]),
        'entropy_coef':   trial.suggest_categorical('entropy_coef', [0.0, 0.01, 0.05]),
        'value_coef':     trial.suggest_categorical('value_coef', [0.5, 1.0]),
        'rollout_length': trial.suggest_categorical('rollout_length', [512, 1024, 2048]),
        'update_epochs':  trial.suggest_categorical('update_epochs', [4, 10]),
        'minibatch_size': trial.suggest_categorical('minibatch_size', [64, 128, 256]),
        'frame_stack':    trial.suggest_categorical('frame_stack', [1, 2, 4]),
        'hidden1':        trial.suggest_categorical('hidden1', [64, 128, 256]),
        'hidden2':        trial.suggest_categorical('hidden2', [64, 128, 256]),
    }


# Sensible "untuned" configurations used as trial 0, so the search has a clear
# baseline reference inside it (the Lab 6 approach). Param names must match the
# search space exactly: DQN uses 'net_arch', PPO uses 'hidden1'/'hidden2'.
BASELINE_DDQN_PARAMS = {
    'net_arch': 128,
    'lr': 1e-3,
    'exploration_fraction': 0.10,
    'epsilon_min': 0.05,
    'batch_size': 128,
    'buffer_size': 50_000,
    'target_update_freq': 250,
    'gradient_steps': 1,
    'train_freq': 4,
    'n_step': 3,
    'frame_stack': 4,
}

BASELINE_PPO_PARAMS = {
    'lr': 3e-4,
    'gae_lambda': 0.95,
    'clip_eps': 0.2,
    'entropy_coef': 0.01,
    'value_coef': 0.5,
    'rollout_length': 1024,
    'update_epochs': 10,
    'minibatch_size': 128,
    'frame_stack': 4,
    'hidden1': 128,
    'hidden2': 128,
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
    verbose=False,
    ckpt_eval_episodes=200,
    baseline=True,
    baseline_params=None,
):
    """
    Tune Double DQN or PPO with Optuna. Each trial trains for n_episodes_tune
    with best-checkpoint selection ON, so the agent scored by Optuna is the
    trial's best checkpoint (Lab 6 approach), not its final weights.

    For metric='combined' uses NSGA-II over (survival_rate, mean_return) with a
    reduced population so the search actually evolves within a small budget.

    Returns
    -------
    (best_params, study, best_hist)
        best_params : dict ready to plug into train_dqn / train_ppo.
        study       : the Optuna study.
        best_hist   : the winning trial's training history; best_hist['agent']
                      is its best-checkpoint policy, reusable without retraining.
    """
    if not _HAS_OPTUNA:
        raise ImportError("optuna is not installed. `pip install optuna`.")

    a = algo.lower().replace(' ', '_').replace('-', '_')
    if a not in ('ddqn', 'double_dqn', 'double', 'ppo'):
        raise ValueError(f"Unknown algo {algo!r}. Choose 'ddqn', 'double_dqn' or 'ppo'.")

    if eval_seeds is None:
        eval_seeds = [seed]

    # Keep each trial's training history (and its best-checkpoint agent) so the
    # winning trial's policy can be reused directly, without retraining.
    trial_hists = {}

    def objective(trial):
        if a in ('ddqn', 'double_dqn', 'double'):
            params = _suggest_dqn_params(trial, double=True)
            hist = train_dqn(n_episodes=n_episodes_tune, double=True, seed=seed,
                             device=device, select_best=True,
                             ckpt_eval_episodes=ckpt_eval_episodes, **params)
        else:
            params = _suggest_ppo_params(trial)
            hist = train_ppo(n_episodes=n_episodes_tune, seed=seed, device=device,
                             select_best=True, ckpt_eval_episodes=ckpt_eval_episodes,
                             **params)

        # Free the large replay buffer (not needed after training) and keep the
        # trial's history + best-checkpoint agent for possible reuse.
        ag = hist['agent']
        if getattr(ag, 'replay_buffer', None) is not None:
            ag.replay_buffer.buffer.clear()
        trial_hists[trial.number] = hist

        seed_summaries = []
        for eval_seed in eval_seeds:
            m = evaluate_policy_b(hist['agent'], n_episodes=eval_episodes, seed=eval_seed)
            seed_summaries.append(m.summary())

        survival_rates = np.array([s['survival_rate'] for s in seed_summaries], dtype=float)
        mean_returns   = np.array([s['mean_return']   for s in seed_summaries], dtype=float)

        survival_rate_mean = float(survival_rates.mean())
        mean_return_mean   = float(mean_returns.mean())
        if len(eval_seeds) > 1:
            survival_rate_std = float(survival_rates.std(ddof=1))
            mean_return_std   = float(mean_returns.std(ddof=1))
        else:
            survival_rate_std = 0.0
            mean_return_std   = 0.0

        trial.set_user_attr("survival_rate_mean", survival_rate_mean)
        trial.set_user_attr("survival_rate_std", survival_rate_std)
        trial.set_user_attr("mean_return_mean", mean_return_mean)
        trial.set_user_attr("mean_return_std", mean_return_std)
        trial.set_user_attr("best_ckpt_ep", hist.get('best_ep'))
        trial.set_user_attr("eval_seeds", list(eval_seeds))
        trial.set_user_attr("eval_episodes_per_seed", eval_episodes)

        if metric == 'survival':
            return survival_rate_mean
        elif metric == 'combined':
            return survival_rate_mean, mean_return_mean
        else:
            return mean_return_mean

    if metric == 'combined':
        study = optuna.create_study(
            directions=['maximize', 'maximize'],
            sampler=optuna.samplers.NSGAIISampler(population_size=max(4, n_trials // 2), seed=seed),
        )
    else:
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=seed),
        )

    # Trial 0 = a sensible untuned baseline, so we can see whether the search
    # actually beats a reasonable default config.
    if baseline:
        bp = baseline_params
        if bp is None:
            bp = (BASELINE_DDQN_PARAMS if a in ('ddqn', 'double_dqn', 'double')
                  else BASELINE_PPO_PARAMS)
        study.enqueue_trial(dict(bp))

    study.optimize(objective, n_trials=n_trials, show_progress_bar=verbose)

    if metric == 'combined':
        best_trial = max(study.best_trials, key=lambda t: (t.values[0], t.values[1]))
    else:
        best_trial = study.best_trial

    best = dict(best_trial.params)
    if 'net_arch' in best:
        arch = best.pop('net_arch')
        best['hidden1'] = arch
        best['hidden2'] = arch

    # The best trial's full history; best_hist['agent'] is its best-checkpoint
    # policy, ready to be saved/used directly without retraining.
    best_hist = trial_hists.get(best_trial.number)

    return best, study, best_hist
