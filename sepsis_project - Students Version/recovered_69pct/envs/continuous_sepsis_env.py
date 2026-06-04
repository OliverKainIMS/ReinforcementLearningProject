"""
continuous_sepsis_env.py
========================
Continuous-observation wrapper for the ICU-Sepsis gymnasium environment.

WHAT THIS IS
------------
The standard ICU-Sepsis-v2 environment returns a discrete integer (0–715)
as the observation, the index of the patient's state in the MDP table.
This makes tabular RL methods like Q-learning technically feasible (the
Q-table has a finite, manageable number of rows).

This wrapper replaces that integer with the 47-dimensional normalised
physiological feature vector that corresponds to that state's cluster centre
in the original MIMIC-III patient data. The underlying MDP dynamics
(transitions, rewards) are IDENTICAL, only the observation changes.

WHY THIS MATTERS (Config B motivation)
--------------------------------------
With continuous observations:
  - A tabular Q-table would need one entry per unique float vector.
  - Since observations are real-valued, the agent never sees the exact same
    vector twice, the table would be infinitely large.


THE 47 FEATURES
---------------
These are the normalised physiological variables from Komorowski et al.
(2018), AI Clinician, used to cluster the 17,000+ MIMIC-III patient states
into 714 non-terminal states. They include vital signs, lab values, and
administered treatments, the real measurements a clinician sees.

USAGE
-----
    from continuous_sepsis_env import ContinuousICUSepsisEnv

    # Default (reproduces paper results):
    env = ContinuousICUSepsisEnv()

    # With difficulty settings — pass the output of make_sepsis_env():
    from <your_notebook_helpers> import make_sepsis_env
    env = ContinuousICUSepsisEnv(params=make_sepsis_env(sofa_bias=4.0, lam=0.02))

    obs, info = env.reset(seed=42)
    # obs.shape == (47,), obs.dtype == float32

    obs, reward, terminated, truncated, info = env.step(action)  # action in 0..24

INSTALL DEPENDENCIES
--------------------
    pip install icu-sepsis stable-baselines3 torch gymnasium
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

#  47 physiological feature names
FEATURE_NAMES = [
    'Albumin', 'ALP', 'ALT', 'AST', 'Bilirubin', 'BUN',
    'Chloride', 'Creatinine', 'Diastolic_BP', 'Fibrinogen',
    'FiO2', 'GCS', 'Glucose', 'HCO3', 'Hemoglobin', 'INR',
    'Lactate', 'MgSO4', 'Mean_BP', 'Potassium', 'PTT',
    'PaCO2', 'PaO2', 'Platelets', 'RBC', 'SOFA', 'Sodium',
    'Systolic_BP', 'Temperature', 'Troponin', 'Urine_Output',
    'WBC', 'Weight', 'Ca', 'Heparin', 'HR', 'Na_Bicarb',
    'Insulin', 'Norepinephrine', 'Propofol', 'Vasopressin',
    'Dopamine', 'Phenylephrine', 'SpO2', 'Milrinone', 'RespRate', 'Age'
]


class ContinuousICUSepsisEnv(gym.Env):
    """
    Continuous-observation ICU-Sepsis environment.

    Observation space : Box(47,) — normalised physiological feature vector
    Action space      : Discrete(25) — vasopressor × IV fluid dose levels
                        action = vasopressor_level * 5 + fluid_level
                        levels 0..4 map to [none, low, medium, high, very high]
    Reward            : +1.0 at survival, 0.0 at death, 0.0 intermediate
                        (modified if a custom params with lam > 0 is passed)
    Discount          : γ = 1.0 (as in the ICU-Sepsis paper)

    Terminal states   : 714 (survived), 713 (died)
    """

    # Environment constants
    FEATURE_NAMES: list = FEATURE_NAMES
    N_FEATURES: int = 47
    N_ACTIONS: int = 25
    STATE_SURVIVED: int = 714
    STATE_DIED: int = 713
    N_STATES: int = 716

    metadata = {'render_modes': []}

    def __init__(self, params=None, obs_noise_std: float = 0.0):
        """
        Args:
            params (MDPParameters or gymnasium.Env, optional):
                Custom environment configuration produced by make_sepsis_env().
                Pass this to apply difficulty settings (SOFA-biased initial
                state distribution, treatment intensity penalty) to the
                continuous wrapper. When None, uses the default ICU-Sepsis-v2
                parameters (reproduces the paper benchmarks).

                Accepted types:
                  - None                 default env (paper settings)
                  - gymnasium.Env        a pre-built env from make_sepsis_env()
                  - MDPParameters        raw params object

                Example:
                    env = ContinuousICUSepsisEnv(
                        params=make_sepsis_env(sofa_bias=4.0, lam=0.02)
                    )

            obs_noise_std (float):
                Standard deviation of Gaussian noise added to observations.
                0.0 (default) = clean observations (standard eval).
                0.05 = small perturbation (robustness eval).
                Use this to test whether your agent generalises or memorises.
        """
        super().__init__()
        self.obs_noise_std = obs_noise_std

        import icu_sepsis  # registers the Sepsis/ICU-Sepsis-v2 environment

        #  Resolve the underlying raw env from whatever params type is given
        if params is None:
            raw = gym.make('Sepsis/ICU-Sepsis-v2').unwrapped
        elif hasattr(params, '_tx_mat'):
            # ICUSepsisEnv returned directly from make_sepsis_env()
            # already the raw env — use it directly, no unwrapping needed
            raw = params
        elif isinstance(params, gym.Env):
            raw = params.unwrapped
        else:
            raw = gym.make('Sepsis/ICU-Sepsis-v2', params=params).unwrapped

        self._raw = raw
        self._levels = raw._action_levels  # 5

        # Cluster centres: shape (716, 47), row i = continuous obs for state i
        # Preserved through MDPParameters.create() always available.
        self._cluster_centers: np.ndarray = raw._state_cluster_centers.astype(np.float32)

        # Gymnasium spaces
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.N_FEATURES,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self.N_ACTIONS)

    #  Internal helpers 

    def _state_to_obs(self, state_idx: int) -> np.ndarray:
        """Return the continuous feature vector for a given state index."""
        obs = self._cluster_centers[state_idx].copy()
        if self.obs_noise_std > 0:
            noise = np.random.normal(0.0, self.obs_noise_std, size=obs.shape)
            obs = (obs + noise.astype(np.float32))
        return obs

    def _flat_to_multi_action(self, action: int) -> np.ndarray:
        """
        Convert a flat Discrete(25) action to MultiDiscrete([5, 5]).
        The underlying env expects [vasopressor_level, fluid_level].
        """
        return np.array([action // self._levels, action % self._levels])

    #  Gymnasium API 

    def reset(self, seed: int = None, **kwargs):
        """
        Reset the environment and return the initial continuous observation.

        Args:
            seed (int, optional): Random seed for reproducibility.

        Returns:
            obs (np.ndarray): Shape (47,), normalised physiological features.
            info (dict): Additional info from the underlying environment.
        """
        if seed is not None:
            self._raw.np_random, _ = gym.utils.seeding.np_random(seed)
        _, info = self._raw.reset()
        state_idx = int(self._raw._current_state)
        return self._state_to_obs(state_idx), info

    def step(self, action: int):
        """
        Take one step in the environment.

        Args:
            action (int): Integer in [0, 24].
                          action = vasopressor_level * 5 + fluid_level

        Returns:
            obs        (np.ndarray): Shape (47,), next state features.
            reward     (float)     : +1.0 (survival), 0.0 (death or ongoing).
                                     If lam > 0 was set via make_sepsis_env(),
                                     intermediate steps carry a treatment
                                     intensity penalty.
            terminated (bool)      : True if patient died or survived.
            truncated  (bool)      : True if max episode length reached.
            info       (dict)      : Additional info.
        """
        multi_action = self._flat_to_multi_action(int(action))
        _, reward, terminated, truncated, info = self._raw.step(multi_action)
        state_idx = int(self._raw._current_state)
        obs = self._state_to_obs(state_idx)
        return obs, float(reward), terminated, truncated, info

    def close(self):
        self._raw.close()

    #  Convenience 

    def decode_action(self, action: int) -> str:
        """Human-readable description of an action."""
        vaso = action // self._levels
        fluid = action % self._levels
        levels = ['none', 'low', 'medium', 'high', 'very high']
        return f'vasopressor={levels[vaso]}, IV_fluid={levels[fluid]}'

    def get_feature_names(self) -> list:
        """Return the list of 47 physiological feature names."""
        return self.FEATURE_NAMES.copy()


#  Quick sanity check (run this file directly to verify installation) 
if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')

    print('=' * 55)
    print('ContinuousICUSepsisEnv — sanity check')
    print('=' * 55)

    #  Default env 
    env = ContinuousICUSepsisEnv()
    obs, info = env.reset(seed=42)

    print(f'Observation space : {env.observation_space}')
    print(f'Action space      : {env.action_space}')
    print(f'Initial obs shape : {obs.shape}')
    print(f'Initial obs dtype : {obs.dtype}')
    print(f'Feature names     : {env.FEATURE_NAMES[:5]} ... (47 total)')
    print()

    # One episode with random actions
    total_r, steps, done = 0.0, 0, False
    while not done:
        action = env.action_space.sample()
        obs, r, te, tr, _ = env.step(action)
        total_r += r
        steps += 1
        done = te or tr
    print(f'Random episode: {steps} steps, return={total_r:.1f} '
          f'({"survived" if total_r > 0 else "died"})')

    # 500-episode survival rate check
    np.random.seed(0)
    returns = []
    for _ in range(500):
        env.reset(seed=np.random.randint(100000))
        total_r, done = 0.0, False
        while not done:
            obs, r, te, tr, _ = env.step(env.action_space.sample())
            total_r += r
            done = te or tr
        returns.append(total_r)

    mean_return = np.mean(returns)
    print(f'Random agent (500 ep): mean return = {mean_return:.4f}')
    env.close()

    #  Hard env (sofa_bias + intensity penalty) 
    print()
    print('Testing with make_sepsis_env(sofa_bias=5.0, lam=0.02) ...')

    import icu_sepsis
    from icu_sepsis.utils.io import MDPParameters

    base = gym.make('Sepsis/ICU-Sepsis-v2')
    raw  = base.unwrapped
    P    = raw._tx_mat
    R_h  = raw._r_mat.copy()
    d0   = raw._d_0.copy()
    sofa = raw._sofa_scores.flatten()
    S, A, _ = P.shape

    INTENSITY = np.array([(a // 5 + a % 5) / 8.0 for a in range(A)])
    for a in range(A):
        R_h[:, a, :] -= 0.02 * INTENSITY[a]

    weight = np.exp(sofa[:714] / 4.0) * (d0[:714] > 0)
    weight /= weight.sum()
    d0_h = np.zeros(S); d0_h[:714] = weight

    hard_params = MDPParameters.create(
        tx_mat=P, r_mat=R_h, d_0=d0_h,
        expert_policy=raw._expert_policy,
        admissible_actions=raw._admissible_actions,
        state_cluster_centers=raw._state_cluster_centers,
        sofa_scores=raw._sofa_scores,
        metadata=raw._metadata,
    )
    base.close()
    hard_gym_env = gym.make('Sepsis/ICU-Sepsis-v2', params=hard_params)

    hard_env = ContinuousICUSepsisEnv(params=hard_gym_env)
    np.random.seed(0)
    hard_returns = []
    for _ in range(500):
        hard_env.reset(seed=np.random.randint(100000))
        total_r, done = 0.0, False
        while not done:
            obs, r, te, tr, _ = hard_env.step(hard_env.action_space.sample())
            total_r += r
            done = te or tr
        hard_returns.append(total_r)

    print(f'Env random agent (500 ep): mean return = {np.mean(hard_returns):.4f}')
    hard_env.close()

    print()
    print('Action decoding example:')
    for a in [0, 6, 12, 18, 24]:
        print(f'  action {a:2d} → {env.decode_action(a)}')

    print()
