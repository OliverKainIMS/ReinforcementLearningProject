"""
wrappers.py — Config B Clinical Reality Wrappers
================================================
Three orthogonal failure modes injected into the ICU-Sepsis environment,
plus the factory function.

Contents
--------
  EpisodicNoisyObsEnv      – episodic monitor malfunction
  EpisodicMissingObsEnv    – episodic missing lab values
  AcuteEventEnv            – rare sudden patient death
  make_clinical_env()      – compose all wrappers into one env
"""

import numpy as np
import gymnasium as gym

from envs.env_setup import make_sepsis_env, SOFA_BIAS, LAM


#  Wrapper 1: Episodic Observation Noise 

class EpisodicNoisyObsEnv(gym.Wrapper):
    """
    Occasionally a monitoring session is noisy, not every reading.

    With probability `malfunction_prob`, this entire episode's observations
    are corrupted by Gaussian noise. Models:
        - Equipment calibration drift
        - Sensor placement issues
        - Monitor malfunction during a shift

    The agent receives clean observations (1 - malfunction_prob) of the time
    and degraded observations the rest. It must learn to be robust to both.

    Args:
        malfunction_prob : probability this episode has noisy readings (default 0.15)
        noise_std        : std of Gaussian noise when active (default 0.10)

    info keys added:
        'noisy_episode' : bool  whether this episode has noise active
    """

    def __init__(self, env, malfunction_prob: float = 0.15, noise_std: float = 0.10):
        super().__init__(env)
        self.malfunction_prob = malfunction_prob
        self.noise_std        = noise_std
        self.episode_noisy    = False

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.episode_noisy = np.random.rand() < self.malfunction_prob
        if self.episode_noisy:
            obs = (obs + np.random.normal(0, self.noise_std, obs.shape)).astype(np.float32)
        info['noisy_episode'] = self.episode_noisy
        return obs, info

    def step(self, action):
        obs, r, te, tr, info = self.env.step(action)
        if self.episode_noisy:
            obs = (obs + np.random.normal(0, self.noise_std, obs.shape)).astype(np.float32)
        info['noisy_episode'] = self.episode_noisy
        return obs, r, te, tr, info


#  Wrapper 2: Episodic Missing Observations 

class EpisodicMissingObsEnv(gym.Wrapper):
    """
    Occasionally specific lab measurements are unavailable for a full episode.

    With probability `missing_prob`, a fixed subset of features is zeroed out
    for the entire episode. The same features are missing throughout. Models:
        - Lab analyser downtime
        - Missing chart entries at admission
        - Equipment failure for a specific measurement type

    The agent must learn to make good decisions with incomplete information.

    Args:
        missing_prob : probability this episode has missing features (default 0.15)
        n_missing    : number of features unavailable when active (default 4)

    info keys added:
        'missing_features' : array of missing feature indices, or None
    """

    def __init__(self, env, missing_prob: float = 0.15, n_missing: int = 4):
        super().__init__(env)
        self.missing_prob = missing_prob
        self.n_missing    = n_missing
        self.missing_mask = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if np.random.rand() < self.missing_prob:
            self.missing_mask = np.random.choice(obs.shape[0], self.n_missing, replace=False)
            obs = obs.copy()
            obs[self.missing_mask] = 0.0
        else:
            self.missing_mask = None
        info['missing_features'] = self.missing_mask
        return obs, info

    def step(self, action):
        obs, r, te, tr, info = self.env.step(action)
        if self.missing_mask is not None:
            obs = obs.copy()
            obs[self.missing_mask] = 0.0
        info['missing_features'] = self.missing_mask
        return obs, r, te, tr, info


#  Wrapper 3: Acute Clinical Events 

class AcuteEventEnv(gym.Wrapper):
    """
    With very low probability each step, the patient suffers an acute
    unrecoverable event and dies, regardless of treatment.

    This is NOT a reward penalty. The episode terminates immediately with
    the death outcome. The agent did everything correctly. The patient still
    died. This is irreducible stochasticity. Models:
        - Massive pulmonary embolism
        - Sudden cardiac arrest
        - Catastrophic secondary organ failure

    The event_prob is deliberately very low (default 0.01):
        Over a 9-step episode → ~9% chance of at least one event.

    Args:
        event_prob : probability of acute event at each step (default 0.01)

    info keys added:
        'acute_event' : bool  whether this step triggered an acute event
    """

    def __init__(self, env, event_prob: float = 0.01):
        super().__init__(env)
        self.event_prob = event_prob

    def step(self, action):
        obs, r, te, tr, info = self.env.step(action)
        # Only fire during ongoing episodes not when already terminal
        if not (te or tr) and np.random.rand() < self.event_prob:
            info['acute_event'] = True
            return obs, 0.0, True, False, info
        info['acute_event'] = False
        return obs, r, te, tr, info


#  Factory: compose all three wrappers 

def make_clinical_env(
    sofa_bias=SOFA_BIAS, lam=LAM,
    malfunction_prob=0.15, noise_std=0.10,
    missing_prob=0.15,    n_missing=4,
    event_prob=0.01,
):
    """
    Full clinical environment: hard config + all three reality wrappers.

    Wrapper order:
        ContinuousICUSepsisEnv          ← base: 47-dim obs, discrete(25) actions
        → EpisodicNoisyObsEnv           ← occasional monitor malfunction
        → EpisodicMissingObsEnv         ← occasional missing lab values
        → AcuteEventEnv                 ← rare sudden patient death

    This is the REQUIRED environment for Config B training and evaluation.
    You may adjust wrapper parameters for sensitivity analysis, but all
    main results must use the defaults above.
    """
    from envs.continuous_sepsis_env import ContinuousICUSepsisEnv

    base = ContinuousICUSepsisEnv(params=make_sepsis_env(sofa_bias=sofa_bias, lam=lam))
    env  = EpisodicNoisyObsEnv(base,  malfunction_prob=malfunction_prob, noise_std=noise_std)
    env  = EpisodicMissingObsEnv(env, missing_prob=missing_prob,         n_missing=n_missing)
    env  = AcuteEventEnv(env,          event_prob=event_prob)
    return env


