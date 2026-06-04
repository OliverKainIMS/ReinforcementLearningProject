"""
env_setup.py — Environment Setup & Shared Helpers
==================================================
Constants, the ICU-Sepsis environment factory, and the plotting / evaluation
helpers shared across all Config A algorithms.

Contents
--------
  Constants                – ENV_ID, N_STATES, N_ACTIONS, GAMMA, …
  make_sepsis_env()        – build the hard-config tabular Sepsis env
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gymnasium as gym
import icu_sepsis  # registers Sepsis/ICU-Sepsis-v2  # noqa: F401


#  ICU-Sepsis constants 

ENV_ID         = 'Sepsis/ICU-Sepsis-v2'
N_STATES       = 716   # 714 clinical + 2 terminal
N_ACTIONS      = 25    # 5 vasopressor levels × 5 IV-fluid dose levels
STATE_SURVIVED = 714
STATE_DIED     = 713
GAMMA          = 1.0   # ICU-Sepsis paper convention no time discounting

# Treatment intensity per action (normalised 0–1)
# action_idx = vaso_level * 5 + fluid_level  →  intensity = (vaso + fluid) / 8
INTENSITY = np.array([(a // 5 + a % 5) / 8.0 for a in range(N_ACTIONS)])

# Fixed configuration 
SOFA_BIAS = 5.0   # sicker patient cohort (mean SOFA ~8.8)
LAM       = 0.02  # treatment intensity penalty (parsimony pressure)


#  Environment factory 

def make_sepsis_env(sofa_bias: float = SOFA_BIAS, lam: float = LAM):
    """
    Build the ICU-Sepsis-v2 environment with configuration:

        sofa_bias : shifts the initial-state distribution toward sicker patients
                    (higher SOFA score at episode start). Default 5.0.
        lam       : treatment intensity penalty subtracted from each reward step.
                    Encourages parsimonious treatment. Default 0.02.


    Returns a gymnasium env with the modified reward and initial-state matrices.
    """
    base = gym.make(ENV_ID)
    raw  = base.unwrapped
    P    = raw._tx_mat
    S, A, _ = P.shape

    # Apply treatment intensity penalty
    R_new = raw._r_mat.copy()
    if lam > 0.0:
        for a in range(A):
            R_new[:, a, :] -= lam * INTENSITY[a]
        print(f'make_sepsis_env | lam={lam} → intensity penalty active')

    # Shift initial-state distribution toward sicker patients
    sofa   = raw._sofa_scores.flatten()
    d0_new = raw._d_0.copy()
    if sofa_bias is not None:
        sofa_clinical = sofa[:S - 2]
        weight  = np.exp(sofa_clinical / sofa_bias)
        weight *= (d0_new[:S - 2] > 0)
        weight /= weight.sum()
        d0_new         = np.zeros(S)
        d0_new[:S - 2] = weight
        mean_sofa = float(np.average(sofa_clinical, weights=weight))
        print(f'make_sepsis_env | sofa_bias={sofa_bias} → mean start SOFA: {mean_sofa:.2f}')

    raw._r_mat = R_new
    raw._d_0   = d0_new

    return base

