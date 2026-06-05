"""
run_multiseed.py
================
Train ONE agent (DDQN or PPO) on a single training seed, then evaluate it on the
held-out test seeds. Designed to be launched many times in parallel (one process
per algo+seed) by run_all.py.

For each run it records:
  * val_*  -> the best-checkpoint score on the validation seed (987654). This is
             what we use later to pick the "best of 5" for the qualitative plots,
             WITHOUT touching the test seeds (no leakage).
  * test_* -> evaluate_policy_multiseed on EVAL_SEEDS (the reporting seeds). The
             mean/std across the 5 TRAINING seeds is computed later in aggregate.py.

Reuses the recovered Optuna hyperparameters (no re-tuning).
"""

import argparse
import json
import os
import time
import warnings

warnings.filterwarnings("ignore")

import torch

# Each run is ~1 core of real work (small nets), so pin to 1 thread and rely on
# running many processes in parallel instead. Keeps the 12-core machine from
# oversubscribing when several runs go at once.
torch.set_num_threads(1)

import rl_configB_functions as R

# --- Recovered Optuna hyperparameters (same as the notebook, no re-tuning) ---
BEST_PARAMS = {
    "ddqn": {
        "lr": 0.000267775333103133,
        "exploration_fraction": 0.05,
        "epsilon_min": 0.05,
        "batch_size": 128,
        "buffer_size": 100000,
        "target_update_freq": 50,
        "gradient_steps": 2,
        "hidden1": 128,
        "hidden2": 128,
    },
    "ppo": {
        "lr": 0.00010900492537293822,
        "gae_lambda": 0.9,
        "clip_eps": 0.1,
        "entropy_coef": 0.05,
        "value_coef": 0.5,
        "rollout_length": 512,
        "update_epochs": 4,
        "minibatch_size": 256,
        "hidden1": 128,
        "hidden2": 128,
    },
}

# Reporting seeds (identical to the random baseline, for a fair comparison).
EVAL_SEEDS = [101, 202, 303, 404, 505]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", required=True, choices=["ddqn", "ppo"])
    ap.add_argument("--seed", type=int, required=True, help="training seed")
    ap.add_argument("--episodes", type=int, default=50000, help="training episodes")
    ap.add_argument("--eval-episodes", type=int, default=50000,
                    help="evaluation episodes PER test seed")
    ap.add_argument("--outdir", default="multiseed")
    args = ap.parse_args()

    ckpt_dir = os.path.join(args.outdir, "checkpoints")
    res_dir = os.path.join(args.outdir, "results")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    tag = f"{args.algo}_seed{args.seed}"
    print(f"[{tag}] starting | episodes={args.episodes} eval_episodes={args.eval_episodes}",
          flush=True)

    t0 = time.time()
    if args.algo == "ddqn":
        hist = R.train_dqn(
            n_episodes=args.episodes, double=True, seed=args.seed,
            verbose=True, log_every=max(1, args.episodes // 10),
            **BEST_PARAMS["ddqn"],
        )
        agent = hist["agent"]
        state = agent.online_net.state_dict()
    else:
        hist = R.train_ppo(
            n_episodes=args.episodes, seed=args.seed,
            verbose=True, log_every=max(1, args.episodes // 10),
            **BEST_PARAMS["ppo"],
        )
        agent = hist["agent"]
        state = agent.net.state_dict()
    train_time = time.time() - t0

    # Validation score of the selected best checkpoint (survival, return) on the
    # held-out validation seed 987654. Used ONLY for best-of-5 selection.
    val_surv, val_ret = hist["best_checkpoint_score"]

    # Test evaluation on the reporting seeds.
    t1 = time.time()
    res = R.evaluate_policy_multiseed(
        agent, eval_seeds=EVAL_SEEDS, n_episodes=args.eval_episodes,
    )
    eval_time = time.time() - t1
    s = res["summary"]

    ckpt_path = os.path.join(ckpt_dir, f"{tag}.pt")
    torch.save(state, ckpt_path)

    out = {
        "algo": args.algo,
        "seed": args.seed,
        "episodes": args.episodes,
        "eval_episodes": args.eval_episodes,
        # --- validation (for best-of-5 selection, no leakage) ---
        "val_survival": float(val_surv),
        "val_return": float(val_ret),
        "best_checkpoint_episode": hist.get("best_checkpoint_episode"),
        # --- test (reporting seeds) ---
        "test_mean_return": s["mean_return"],
        "test_mean_return_std": s["mean_return_std"],
        "test_survival_rate": s["survival_rate"],
        "test_survival_rate_std": s["survival_rate_std"],
        "test_mean_length": s["mean_length"],
        "per_seed": res["per_seed"].to_dict(orient="records"),
        # --- bookkeeping ---
        "train_time_sec": train_time,
        "eval_time_sec": eval_time,
        "checkpoint": ckpt_path,
    }
    out_path = os.path.join(res_dir, f"{tag}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"[{tag}] DONE | test survival={s['survival_rate']:.4f} "
          f"return={s['mean_return']:.4f} | val survival={val_surv:.4f} "
          f"| train {train_time/60:.1f} min, eval {eval_time/60:.1f} min",
          flush=True)


if __name__ == "__main__":
    main()
