"""
build_configB_notebook.py
=========================
Generate `02_config_B_continuous.ipynb` for ICU-Sepsis Config B.

The notebook is thin: every reusable function/class lives in
`rl_configB_continuous_functions.py`; here we only assemble the markdown +
code cells that drive the full Config B workflow (18 sections).

Usage
-----
    python build_configB_notebook.py

This uses `nbformat` when available (the requested path). If `nbformat` is not
installed it falls back to writing the equivalent notebook JSON with the
standard library so the notebook is always produced.
"""

OUTPUT = "02_config_B_continuous.ipynb"


# Each entry is ("markdown" | "code", source_string).
CELLS = [
    # ── Title ────────────────────────────────────────────────────────────────
    ("markdown", """# ICU-Sepsis — Configuration B (Continuous Observations)

**DQN vs. Double DQN vs. PPO** on the continuous-observation ICU-Sepsis
environment (`Box(47,)` observations, `Discrete(25)` actions = 5 vasopressor ×
5 IV-fluid levels), with optional clinical-reality wrappers (noisy obs, missing
obs, acute events).

All reusable logic lives in `rl_configB_continuous_functions.py`; this notebook
just orchestrates the workflow.

**Sections**
1. Imports and setup
2. Global configuration
3. Environment creation and validation
4. Random & conservative baselines
5. Optuna tuning — DQN
6. Optuna tuning — Double DQN
7. Optuna tuning — PPO
8. Final DQN training (up to 50,000 episodes)
9. Final Double DQN training (up to 50,000 episodes)
10. Final PPO training (up to 50,000 episodes)
11. Evaluation — clean Config B
12. Evaluation — noisy observations
13. Evaluation — missing observations
14. Evaluation — acute clinical events
15. Evaluation — combined wrappers (optional)
16. Learning-curve plots
17. Results table
18. Final summary"""),

    # ── 1. Imports and setup ─────────────────────────────────────────────────
    ("markdown", """## 1. Imports and setup

We import the helper module and seed all RNGs for reproducibility."""),
    ("code", """import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt

from rl_configB_continuous_functions import (
    set_seed, SEED,
    make_configB_env, validate_env,
    decode_action, action_to_str, treatment_intensity,
    RandomPolicy, ConservativePolicy,
    train_dqn, train_ppo,
    evaluate_agent, clinical_score, run_optuna,
    plot_return_curves, plot_survival_curves,
    summarize_run, build_results_table, save_results, save_checkpoint,
)

set_seed(SEED)"""),

    # ── 2. Global configuration ──────────────────────────────────────────────
    ("markdown", """## 2. Global configuration

`DEBUG_MODE` is the single switch between a quick debugging run and the full
experiment. The final experiments train each agent for **50,000 episodes**
(`N_EPISODES_FINAL`); set `DEBUG_MODE = False` to enable that.

Optuna always tunes on a *much smaller* budget than the final run, and the best
hyperparameters found are then used for the 50,000-episode training."""),
    ("code", """# --- master switch -------------------------------------------------------
DEBUG_MODE = True            # set False for the full 50,000-episode experiments

# --- training horizon ----------------------------------------------------
N_EPISODES_FINAL = 50_000    # final experiment horizon (assignment target)
N_EPISODES_DEBUG = 5_000     # quick smoke-test horizon
N_EPISODES = N_EPISODES_DEBUG if DEBUG_MODE else N_EPISODES_FINAL

# --- evaluation budget ---------------------------------------------------
EVAL_EPISODES = 200 if DEBUG_MODE else 1_000

# --- Optuna tuning budget (small relative to the final run) --------------
RUN_OPTUNA         = True
N_TRIALS           = 3   if DEBUG_MODE else 20
N_EPISODES_TUNE    = 300 if DEBUG_MODE else 500
EVAL_EPISODES_TUNE = 100 if DEBUG_MODE else 200

# --- learning-curve smoothing window ------------------------------------
MA_WINDOW = 100 if DEBUG_MODE else 500

# --- sensible defaults used if Optuna is skipped/unavailable -------------
# DQN / Double DQN: note the explicit replay-buffer size and target-update freq.
# epsilon_decay=0.995 reaches the 0.05 floor after ~600 episodes, ensuring the
# agent exploits its learned values for the majority of training.
DEFAULT_PARAMS = {
    "dqn":  dict(learning_rate=1e-3, gamma=1.0, buffer_size=50_000, batch_size=64,
                 target_update_frequency=200, epsilon_decay=0.995, epsilon_end=0.05),
    "ddqn": dict(learning_rate=1e-3, gamma=1.0, buffer_size=50_000, batch_size=64,
                 target_update_frequency=200, epsilon_decay=0.995, epsilon_end=0.05),
    # PPO has no replay buffer / target network; it exposes its own knobs.
    # rollout_length=256 (debug) gives ~20 steps-per-episode episodes per rollout,
    # so the agent makes many policy updates even in short debug runs.
    "ppo":  dict(learning_rate=3e-4, gamma=1.0, gae_lambda=0.95, clip_eps=0.2,
                 entropy_coef=0.01, value_coef=0.5,
                 rollout_length=256 if DEBUG_MODE else 1024,
                 update_epochs=10, minibatch_size=64),
}
best_params = {k: dict(v) for k, v in DEFAULT_PARAMS.items()}

print(f"DEBUG_MODE={DEBUG_MODE}")
print(f"Training episodes : {N_EPISODES:,}")
print(f"Eval episodes     : {EVAL_EPISODES}")
print(f"Optuna            : run={RUN_OPTUNA}, trials={N_TRIALS}, "
      f"tune_episodes={N_EPISODES_TUNE}")"""),

    # ── 3. Environment creation and validation ───────────────────────────────
    ("markdown", """## 3. Environment creation and validation

We build each evaluation scenario and run a short random rollout to confirm the
observation shape `(47,)` and the `Discrete(25)` action space. We also show how
flat actions decode into (vasopressor, IV-fluid) levels and their treatment
intensity."""),
    ("code", """for sc in ["clean", "noisy", "missing", "acute", "combined"]:
    report = validate_env(lambda sc=sc: make_configB_env(sc), n_steps=10)
    print(f"{sc:9s} -> obs{report['obs_shape']} {report['action_space']} "
          f"rollout_reward={report['rollout_reward']:.3f}")

print()
for a in [0, 6, 12, 18, 24]:
    print(f"action {a:2d}: {action_to_str(a):42s} intensity={treatment_intensity(a):.3f}")"""),

    # ── 4. Baselines ─────────────────────────────────────────────────────────
    ("markdown", """## 4. Random and conservative baselines

Two reference policies, evaluated with the same standalone `evaluate_agent`:
* **Random** — uniform over the 25 actions.
* **Conservative** — always action 0 (no vasopressor, no IV fluid): the
  cheapest policy under the treatment-intensity penalty."""),
    ("code", """def show_eval(name, res):
    print(f"{name:13s} return={res['mean_return']:.3f}  survival={res['survival_rate']:.3f}  "
          f"intensity={res['mean_intensity']:.3f}  vaso={res['mean_vaso']:.2f}  "
          f"fluid={res['mean_fluid']:.2f}")

rand_eval = evaluate_agent(RandomPolicy(seed=SEED), scenario="clean", n_episodes=EVAL_EPISODES)
cons_eval = evaluate_agent(ConservativePolicy(), scenario="clean", n_episodes=EVAL_EPISODES)
show_eval("Random", rand_eval)
show_eval("Conservative", cons_eval)"""),

    # ── 5. Optuna DQN ────────────────────────────────────────────────────────
    ("markdown", """## 5. Optuna tuning — DQN

We tune the DQN-specific knobs (`buffer_size`, `target_update_frequency`,
`batch_size`, `learning_rate`, `gamma`, `epsilon_decay`, `epsilon_end`) on a
small episode budget. The objective is the mean evaluation return (switch to
`metric="clinical"` to optimise survival minus treatment intensity)."""),
    ("code", """if RUN_OPTUNA:
    try:
        bp, study_dqn = run_optuna(
            "dqn", n_trials=N_TRIALS, n_episodes_tune=N_EPISODES_TUNE,
            eval_episodes=EVAL_EPISODES_TUNE, train_scenario="combined",
            eval_scenario="clean", metric="return", seed=SEED,
        )
        best_params["dqn"] = bp
    except ImportError as e:
        print("Optuna unavailable -> using defaults:", e)
print("DQN hyperparameters:", best_params["dqn"])"""),

    # ── 6. Optuna Double DQN ─────────────────────────────────────────────────
    ("markdown", """## 6. Optuna tuning — Double DQN

Same search space as DQN, but training the Double-DQN target (online net
selects the next action, target net evaluates it)."""),
    ("code", """if RUN_OPTUNA:
    try:
        bp, study_ddqn = run_optuna(
            "ddqn", n_trials=N_TRIALS, n_episodes_tune=N_EPISODES_TUNE,
            eval_episodes=EVAL_EPISODES_TUNE, train_scenario="combined",
            eval_scenario="clean", metric="return", seed=SEED,
        )
        best_params["ddqn"] = bp
    except ImportError as e:
        print("Optuna unavailable -> using defaults:", e)
print("Double DQN hyperparameters:", best_params["ddqn"])"""),

    # ── 7. Optuna PPO ────────────────────────────────────────────────────────
    ("markdown", """## 7. Optuna tuning — PPO

PPO has **no replay buffer or target network**. We instead tune its on-policy
knobs: `rollout_length`, `update_epochs`, `minibatch_size`, `clip_eps`,
`entropy_coef`, `value_coef`, `learning_rate`, `gamma`, `gae_lambda`."""),
    ("code", """if RUN_OPTUNA:
    try:
        bp, study_ppo = run_optuna(
            "ppo", n_trials=N_TRIALS, n_episodes_tune=N_EPISODES_TUNE,
            eval_episodes=EVAL_EPISODES_TUNE, train_scenario="combined",
            eval_scenario="clean", metric="return", seed=SEED,
        )
        best_params["ppo"] = bp
    except ImportError as e:
        print("Optuna unavailable -> using defaults:", e)
print("PPO hyperparameters:", best_params["ppo"])"""),

    # ── 8. Final DQN ─────────────────────────────────────────────────────────
    ("markdown", """## 8. Final training — DQN (up to 50,000 episodes)

The replay-buffer size and target-update frequency actually used are logged in
the returned history (`buffer_size`, `target_update_frequency`, per-episode
`buffer_sizes`, `epsilons`, `losses`)."""),
    ("code", """hist_dqn = train_dqn(n_episodes=N_EPISODES, double=False, seed=SEED,
                     verbose=True, ma_window=MA_WINDOW, **best_params["dqn"])
print(f"DQN done | buffer_size={hist_dqn['buffer_size']} "
      f"| target_update_frequency={hist_dqn['target_update_frequency']}")
save_checkpoint(hist_dqn["agent"], "checkpoints/dqn_configB.pt")"""),

    # ── 9. Final Double DQN ──────────────────────────────────────────────────
    ("markdown", """## 9. Final training — Double DQN (up to 50,000 episodes)"""),
    ("code", """hist_ddqn = train_dqn(n_episodes=N_EPISODES, double=True, seed=SEED,
                      verbose=True, ma_window=MA_WINDOW, **best_params["ddqn"])
print("Double DQN done")
save_checkpoint(hist_ddqn["agent"], "checkpoints/ddqn_configB.pt")"""),

    # ── 10. Final PPO ────────────────────────────────────────────────────────
    ("markdown", """## 10. Final training — PPO (up to 50,000 episodes)

PPO logs its policy loss, value loss, entropy and clip fraction per update in
`hist_ppo['ppo_updates']`."""),
    ("code", """hist_ppo = train_ppo(n_episodes=N_EPISODES, seed=SEED,
                     verbose=True, ma_window=MA_WINDOW, **best_params["ppo"])
print("PPO done")
u = hist_ppo["ppo_updates"]
if u["policy_loss"]:
    print(f"last update | policy_loss={u['policy_loss'][-1]:.4f} "
          f"value_loss={u['value_loss'][-1]:.4f} entropy={u['entropy'][-1]:.3f} "
          f"clip_fraction={u['clip_fraction'][-1]:.3f}")
save_checkpoint(hist_ppo["agent"], "checkpoints/ppo_configB.pt")"""),

    # ── 11–15. Evaluation ────────────────────────────────────────────────────
    ("markdown", """## 11–15. Evaluation across scenarios

The evaluation function is **separate** from training. DQN / Double DQN act
greedily (no epsilon); PPO uses its learned policy with `greedy=True` (no
exploration noise). We collect comparison-table rows for every scenario."""),
    ("code", """agents     = {"DQN": hist_dqn["agent"], "Double DQN": hist_ddqn["agent"], "PPO": hist_ppo["agent"]}
params_map = {"DQN": best_params["dqn"], "Double DQN": best_params["ddqn"], "PPO": best_params["ppo"]}
results_rows = []

def eval_scenario(scenario):
    print(f"=== scenario: {scenario} ===")
    for name, ag in agents.items():
        res = evaluate_agent(ag, scenario=scenario, n_episodes=EVAL_EPISODES, seed=SEED)
        results_rows.append(summarize_run(name, res, N_EPISODES, params_map[name], scenario))
        show_eval(name, res)
    print()"""),

    ("markdown", "### 11. Clean Config B environment"),
    ("code", "eval_scenario('clean')"),
    ("markdown", "### 12. Noisy observations"),
    ("code", "eval_scenario('noisy')"),
    ("markdown", "### 13. Missing observations"),
    ("code", "eval_scenario('missing')"),
    ("markdown", "### 14. Acute clinical events"),
    ("code", "eval_scenario('acute')"),
    ("markdown", "### 15. Combined wrappers (optional)"),
    ("code", "eval_scenario('combined')"),

    # ── 16. Learning-curve plots ─────────────────────────────────────────────
    ("markdown", """## 16. Learning-curve plots

Two focused plots, each comparing DQN, Double DQN and PPO on the same figure
over the training episodes (x-axis 0 → `N_EPISODES`):
1. Episode **return**.
2. **Survival rate**."""),
    ("code", """histories = {"DQN": hist_dqn, "Double DQN": hist_ddqn, "PPO": hist_ppo}
plot_return_curves(histories,   window=MA_WINDOW, max_episodes=N_EPISODES,
                   savepath="plots/configB_returns.png")
plot_survival_curves(histories, window=MA_WINDOW, max_episodes=N_EPISODES,
                     savepath="plots/configB_survival.png")"""),

    # ── 17. Results table ────────────────────────────────────────────────────
    ("markdown", """## 17. Results table

One row per (algorithm × evaluation scenario), saved to CSV."""),
    ("code", """table = build_results_table(results_rows)
save_results(table, "plots/configB_results.csv")
table"""),

    # ── 18. Final summary ────────────────────────────────────────────────────
    ("markdown", """## 18. Final summary

Summary of clean-scenario performance and the clinical score
(`survival_rate − 0.5 × treatment_intensity`). With `DEBUG_MODE = False` and the
full 50,000-episode budget, expect the learned agents to clearly exceed the
random and conservative baselines on survival while keeping treatment intensity
moderate."""),
    ("code", """print("Clean-scenario summary (clinical score = survival - 0.5*intensity):\\n")
for name, ag in agents.items():
    res = evaluate_agent(ag, scenario="clean", n_episodes=EVAL_EPISODES, seed=SEED)
    print(f"{name:13s} return={res['mean_return']:.3f}  survival={res['survival_rate']:.3f}  "
          f"intensity={res['mean_intensity']:.3f}  clinical_score={clinical_score(res):.3f}")

print(f"\\nReference baselines (clean):")
print(f"  Random       survival={rand_eval['survival_rate']:.3f}")
print(f"  Conservative survival={cons_eval['survival_rate']:.3f}")"""),
]


def _build_with_nbformat(path):
    import nbformat as nbf
    nb = nbf.v4.new_notebook()
    cells = []
    for kind, src in CELLS:
        if kind == "markdown":
            cells.append(nbf.v4.new_markdown_cell(src))
        else:
            cells.append(nbf.v4.new_code_cell(src))
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    with open(path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)


def _build_with_json(path):
    import json

    def cell(kind, src):
        lines = src.splitlines(keepends=True)
        if kind == "markdown":
            return {"cell_type": "markdown", "metadata": {}, "source": lines}
        return {"cell_type": "code", "metadata": {}, "execution_count": None,
                "outputs": [], "source": lines}

    nb = {
        "cells": [cell(k, s) for k, s in CELLS],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)


if __name__ == "__main__":
    try:
        _build_with_nbformat(OUTPUT)
        print(f"Wrote {OUTPUT} using nbformat ({len(CELLS)} cells).")
    except Exception as exc:  # nbformat missing or any failure -> stdlib fallback
        _build_with_json(OUTPUT)
        print(f"Wrote {OUTPUT} using json fallback ({len(CELLS)} cells). [{exc}]")
