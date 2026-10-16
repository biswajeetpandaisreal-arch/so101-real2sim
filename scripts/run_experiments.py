"""Train and evaluate every condition of the controlled comparison and the
ablations. Resume-safe: a run whose evaluation summary exists is skipped.

  real        sim replays of the recorded demonstrations only          3 seeds
  gen200      + 200 generated demonstrations (method)                  3 seeds
  naive200    + 200 episodes, cube moved, arm motion unchanged         3 seeds
  gen100 / gen25   fewer generated demonstrations (ablation)           1 seed
  nofilter200      generated without the physics filter (ablation)     1 seed

    python scripts/run_experiments.py --parallel 3
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
PLAN = [(c, s) for s in (0, 1, 2) for c in ("real", "gen200", "naive200")] + \
       [("gen100", 0), ("gen25", 0), ("nofilter200", 0)] +        [("reallong", 0), ("gen200long", 0)]   # post-hoc diagnostic: 2x training steps
# Fewer generated demos = the first episodes of gen200 (same seed, so identical to
# generating them separately): its replays plus the first N generated episodes.
SUBSETS = {"gen25": ("gen200", 25), "gen100": ("gen200", 100)}


def dataset_of(cond):
    """(dataset folder, episode indices or None for all)."""
    cond = cond.removesuffix("long")
    if cond not in SUBSETS:
        return cond, None
    base, n = SUBSETS[cond]
    log_file = ROOT / "data" / "datasets" / base / "generation_log.json"
    if not log_file.exists():
        return base, None
    replays = sum(r["kind"] == "replay" and r["success"] for r in json.loads(log_file.read_text()))
    return base, list(range(replays + n))


def train_cmd(cond, seed, steps, batch, save_every):
    ds, eps = dataset_of(cond)
    sel = [f"--dataset.episodes=[{','.join(map(str, eps))}]"] if eps else []
    return [PY, "scripts/train.py", "--policy.type=act", "--policy.device=cuda",
            "--policy.push_to_hub=false", f"--dataset.repo_id=local/so101_sim", *sel,
            f"--dataset.root=data/datasets/{ds}", "--dataset.video_backend=pyav",
            f"--output_dir=outputs/train/{cond}_s{seed}", f"--steps={steps}", f"--batch_size={batch}",
            f"--save_freq={save_every}", "--log_freq=500", "--num_workers=4", f"--seed={seed}",
            "--wandb.enable=false"]


def eval_cmd(cond, seed):
    return [PY, "scripts/evaluate.py", "--run", f"outputs/train/{cond}_s{seed}",
            "--out", f"results/eval/{cond}_s{seed}", "--start-from", "data/datasets/real"]


def job(cond, seed, a):
    """Shell command string: train (unless checkpoints exist), then evaluate."""
    log = ROOT / "outputs" / "logs" / f"run_{cond}_s{seed}.log"
    ck = ROOT / "outputs" / "train" / f"{cond}_s{seed}" / "checkpoints"
    steps = [] if ck.exists() and any(p.name.isdigit() for p in ck.iterdir()) else \
        [train_cmd(cond, seed, a.steps * (2 if cond.endswith("long") else 1), a.batch,
                   a.save_every * (2 if cond.endswith("long") else 1))]
    return steps + [eval_cmd(cond, seed)], log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parallel", type=int, default=3)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--only", default=None, help="comma-separated run names, e.g. real_s0,gen200_s0")
    ap.add_argument("--wait-hours", type=float, default=4.0, help="give up waiting for datasets after this")
    a = ap.parse_args()
    done = lambda c: (ROOT / "data" / "datasets" / dataset_of(c)[0] / "generation_log.json").exists()  # noqa: E731
    finished = lambda c, s: (ROOT / "results" / "eval" / f"{c}_s{s}" / "summary.json").exists()  # noqa: E731
    plan = [(c, s) for c, s in PLAN if not a.only or f"{c}_s{s}" in a.only.split(",")]
    started, running, t_start = set(), [], time.time()
    print(f"plan: {[f'{c}_s{s}' for c, s in plan]}", flush=True)
    while True:
        ready = [(c, s) for c, s in plan if (c, s) not in started and not finished(c, s) and done(c)]
        while ready and len(running) < a.parallel:          # datasets are picked up as they appear
            c, s = ready.pop(0)
            started.add((c, s))
            cmds, log = job(c, s, a)
            log.parent.mkdir(parents=True, exist_ok=True)
            f = open(log, "w")
            script = " && ".join(subprocess.list2cmdline(x) for x in cmds)
            running.append((f"{c}_s{s}", subprocess.Popen(script, shell=True, cwd=ROOT, stdout=f,
                                                          stderr=subprocess.STDOUT), f, time.time()))
            print(f"started {c}_s{s}", flush=True)
        for r in list(running):
            name, p, f, t0 = r
            if p.poll() is not None:
                f.close()
                running.remove(r)
                print(f"finished {name} (exit {p.returncode}, {(time.time() - t0) / 60:.0f} min)", flush=True)
        left = [(c, s) for c, s in plan if (c, s) not in started and not finished(c, s)]
        if not running and (not left or time.time() - t_start > a.wait_hours * 3600):
            print(f"done; never started: {left}", flush=True)
            return
        time.sleep(10)

if __name__ == "__main__":
    main()
