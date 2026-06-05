"""
run_all.py
==========
Driver: launches all (algo x training-seed) runs as independent subprocesses,
limiting concurrency so the 12-core machine is not oversubscribed. Each run is
isolated (its own process), so one crash does not take down the others.

When everything finishes it calls aggregate.py to build the final table.

Usage:
    python run_all.py                      # full run (defaults below)
    python run_all.py --episodes 2000 --eval-episodes 2000 --max-workers 4
"""

import argparse
import itertools
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

PY = sys.executable  # same interpreter that launched this driver


def run_job(job, episodes, eval_episodes, outdir, logdir):
    algo, seed = job
    tag = f"{algo}_seed{seed}"
    log_path = os.path.join(logdir, f"{tag}.log")
    t0 = time.time()
    with open(log_path, "w") as logf:
        proc = subprocess.run(
            [PY, "run_multiseed.py",
             "--algo", algo,
             "--seed", str(seed),
             "--episodes", str(episodes),
             "--eval-episodes", str(eval_episodes),
             "--outdir", outdir],
            stdout=logf, stderr=subprocess.STDOUT,
        )
    return tag, proc.returncode, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--algos", nargs="+", default=["ddqn", "ppo"])
    ap.add_argument("--episodes", type=int, default=50000)
    ap.add_argument("--eval-episodes", type=int, default=50000)
    ap.add_argument("--max-workers", type=int, default=5,
                    help="how many runs in parallel (5 x 2 threads = 10 <= 12 cores)")
    ap.add_argument("--outdir", default="multiseed")
    args = ap.parse_args()

    logdir = os.path.join(args.outdir, "logs")
    os.makedirs(logdir, exist_ok=True)

    jobs = list(itertools.product(args.algos, args.seeds))
    print(f"Launching {len(jobs)} runs | episodes={args.episodes} "
          f"eval_episodes={args.eval_episodes} max_workers={args.max_workers}",
          flush=True)
    print(f"Logs -> {logdir}/<algo>_seed<seed>.log", flush=True)

    t_start = time.time()
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = [ex.submit(run_job, job, args.episodes, args.eval_episodes,
                             args.outdir, logdir) for job in jobs]
        for fut in futures:
            tag, rc, dt = fut.result()
            status = "OK" if rc == 0 else f"FAILED(rc={rc})"
            print(f"[{status}] {tag} ({dt/60:.1f} min)", flush=True)

    print(f"\nAll runs finished in {(time.time()-t_start)/60:.1f} min. Aggregating...",
          flush=True)
    subprocess.run([PY, "aggregate.py", "--outdir", args.outdir])


if __name__ == "__main__":
    main()
