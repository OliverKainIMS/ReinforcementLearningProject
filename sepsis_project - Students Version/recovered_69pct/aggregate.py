"""
aggregate.py
============
Reads all per-run JSONs produced by run_multiseed.py and builds the final
multi-seed comparison.

Two numbers per algorithm, exactly as agreed:
  * mean +/- std  ACROSS the 5 training seeds  -> the honest headline comparison.
  * best-of-5     selected by VALIDATION score (seed 987654), then its TEST
                  score reported -> the single representative agent for the
                  qualitative plots / Andreea's analysis (no test-seed leakage).

Outputs:
  multiseed/results_multiseed.csv      (one row per run)
  multiseed/summary_multiseed.csv      (aggregated table)
  printed table to stdout
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

# Random baseline from the notebook (5 eval seeds x 50k eps). std is across eval
# seeds (random has no training seed) - shown only as a reference row.
RANDOM_BASELINE = {
    "mean_return": 0.5622, "mean_return_std": 0.0025,
    "survival_rate": 0.6547, "survival_rate_std": 0.0024,
}

ALGO_LABEL = {"ddqn": "Double DQN", "ppo": "PPO"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="multiseed")
    args = ap.parse_args()

    res_files = sorted(glob.glob(os.path.join(args.outdir, "results", "*.json")))
    if not res_files:
        print(f"No result JSONs found in {args.outdir}/results/ - nothing to aggregate.")
        return

    runs = [json.load(open(f)) for f in res_files]
    runs_df = pd.DataFrame(runs)
    runs_csv = os.path.join(args.outdir, "results_multiseed.csv")
    keep = ["algo", "seed", "episodes", "eval_episodes", "val_survival", "val_return",
            "best_checkpoint_episode", "test_survival_rate", "test_mean_return",
            "test_mean_length", "train_time_sec", "eval_time_sec"]
    runs_df[ [c for c in keep if c in runs_df.columns] ].to_csv(runs_csv, index=False)

    summary_rows = []

    # Reference random baseline row.
    summary_rows.append({
        "policy": "Random",
        "n_seeds": 0,
        "mean_return_avg": RANDOM_BASELINE["mean_return"],
        "mean_return_std": RANDOM_BASELINE["mean_return_std"],
        "survival_avg": RANDOM_BASELINE["survival_rate"],
        "survival_std": RANDOM_BASELINE["survival_rate_std"],
        "best_seed": np.nan,
        "best_return": RANDOM_BASELINE["mean_return"],
        "best_survival": RANDOM_BASELINE["survival_rate"],
    })

    for algo in ["ddqn", "ppo"]:
        sub = runs_df[runs_df["algo"] == algo]
        if sub.empty:
            continue

        # --- headline: mean +/- std across TRAINING seeds ---
        ret_avg = sub["test_mean_return"].mean()
        ret_std = sub["test_mean_return"].std(ddof=1) if len(sub) > 1 else 0.0
        surv_avg = sub["test_survival_rate"].mean()
        surv_std = sub["test_survival_rate"].std(ddof=1) if len(sub) > 1 else 0.0

        # --- best-of-5 by VALIDATION score (survival first, return as tiebreak) ---
        best_idx = sub.sort_values(["val_survival", "val_return"],
                                   ascending=False).index[0]
        best = sub.loc[best_idx]

        summary_rows.append({
            "policy": ALGO_LABEL[algo],
            "n_seeds": len(sub),
            "mean_return_avg": ret_avg,
            "mean_return_std": ret_std,
            "survival_avg": surv_avg,
            "survival_std": surv_std,
            "best_seed": int(best["seed"]),
            "best_return": float(best["test_mean_return"]),
            "best_survival": float(best["test_survival_rate"]),
        })

    summary = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(args.outdir, "summary_multiseed.csv")
    summary.to_csv(summary_csv, index=False)

    # --- pretty print ---
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    print("\n" + "=" * 78)
    print("MULTI-SEED SUMMARY  (mean +/- std across training seeds  |  best-of-5 by validation)")
    print("=" * 78)
    for _, r in summary.iterrows():
        line = (f"{r['policy']:<12} | "
                f"return {r['mean_return_avg']:.4f} ± {r['mean_return_std']:.4f}"
                f"  survival {r['survival_avg']:.2%} ± {r['survival_std']:.2%}")
        if not np.isnan(r["best_seed"]):
            line += (f"   ||  best(seed {int(r['best_seed'])}): "
                     f"return {r['best_return']:.4f}  survival {r['best_survival']:.2%}")
        print(line)
    print("=" * 78)
    print(f"\nPer-run    -> {runs_csv}")
    print(f"Summary    -> {summary_csv}")


if __name__ == "__main__":
    main()
