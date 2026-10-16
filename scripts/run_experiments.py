"""Train and evaluate every run of the comparison, a few at a time on one GPU.

Resume-safe: a run with an evaluation summary is skipped, and a run with
checkpoints but no summary is only evaluated. Runs start as soon as their
dataset (scripts/generate.py) is complete.

  real          replays of the recorded demonstrations only      3 seeds
  gen200        + 200 generated demonstrations (method)          3 seeds
  naive200      + 200 naive-shift episodes (negative control)    3 seeds
  reallong,     2x training steps (post-hoc check of under-      1 seed
  gen200long    training; not part of the pre-registered plan)
  gen25, gen100, nofilter200   ablations (not run for lack of time; --only)

    python scripts/run_experiments.py --parallel 3
"""

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASETS = ROOT / "data" / "datasets"


@dataclass(frozen=True)
class Condition:
    dataset: str
    steps_factor: int = 1
    generated: int | None = None              # first N generated episodes of the dataset (after its replays)


CONDITIONS = {
    "real": Condition("real"), "gen200": Condition("gen200"), "naive200": Condition("naive200"),
    "reallong": Condition("real", steps_factor=2), "gen200long": Condition("gen200", steps_factor=2),
    "gen25": Condition("gen200", generated=25), "gen100": Condition("gen200", generated=100),
    "nofilter200": Condition("nofilter200"),
}
DEFAULT_PLAN = [f"{c}_s{s}" for s in (0, 1, 2) for c in ("real", "gen200", "naive200")]
DEFAULT_PLAN += ["reallong_s0", "gen200long_s0"]


def episodes_arg(cond: Condition) -> list[str]:
    """--dataset.episodes selecting the replays plus the first N generated episodes (same seed, so
    identical to generating N separately)."""
    if cond.generated is None:
        return []
    log = json.loads((DATASETS / cond.dataset / "generation_log.json").read_text())
    n = sum(r["kind"] == "replay" and r["success"] for r in log) + cond.generated
    return [f"--dataset.episodes=[{','.join(map(str, range(n)))}]"]


def commands(run: str, a) -> list[list[str]]:
    name, seed = run.rsplit("_s", 1)
    cond = CONDITIONS[name]
    out_dir = f"outputs/train/{run}"
    checkpoints = ROOT / out_dir / "checkpoints"
    train = [sys.executable, "scripts/train.py", "--policy.type=act", "--policy.device=cuda",
             "--policy.push_to_hub=false", "--dataset.repo_id=local/so101_sim", *episodes_arg(cond),
             f"--dataset.root=data/datasets/{cond.dataset}", "--dataset.video_backend=pyav", f"--output_dir={out_dir}",
             f"--steps={a.steps * cond.steps_factor}", f"--save_freq={a.save_every * cond.steps_factor}",
             f"--batch_size={a.batch}", "--log_freq=500", "--num_workers=4", f"--seed={seed}", "--wandb.enable=false"]
    evaluate = [sys.executable, "scripts/evaluate.py", "--run", out_dir, "--out", f"results/eval/{run}"]
    trained = checkpoints.exists() and any(p.name.isdigit() for p in checkpoints.iterdir())
    return ([] if trained else [train]) + [evaluate]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--parallel", type=int, default=3, help="concurrent runs (3 fit a 12 GB GPU; 5 do not)")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--only", default=",".join(DEFAULT_PLAN), help="comma-separated runs, e.g. real_s0,gen200_s1")
    ap.add_argument("--wait-hours", type=float, default=4.0, help="stop waiting for missing datasets after this")
    a = ap.parse_args()
    plan = a.only.split(",")
    (ROOT / "outputs" / "logs").mkdir(parents=True, exist_ok=True)
    done = lambda run: (ROOT / "results" / "eval" / run / "summary.json").exists()                      # noqa: E731
    ready = lambda run: (DATASETS / CONDITIONS[run.rsplit("_s", 1)[0]].dataset / "generation_log.json").exists()  # noqa: E731
    started, running, t_start = set(), [], time.time()
    while True:
        for run in [r for r in plan if r not in started and not done(r) and ready(r)]:
            if len(running) >= a.parallel:
                break
            started.add(run)
            log = open(ROOT / "outputs" / "logs" / f"run_{run}.log", "w")
            script = " && ".join(subprocess.list2cmdline(c) for c in commands(run, a))
            running.append((run, subprocess.Popen(script, shell=True, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT),
                            log, time.time()))
            print(f"started {run}", flush=True)
        for entry in list(running):
            run, proc, log, t0 = entry
            if proc.poll() is not None:
                log.close()
                running.remove(entry)
                print(f"finished {run} (exit {proc.returncode}, {(time.time() - t0) / 60:.0f} min)", flush=True)
        waiting = [r for r in plan if r not in started and not done(r)]
        if not running and (not waiting or time.time() - t_start > a.wait_hours * 3600):
            print(f"done; never started: {waiting}", flush=True)
            return
        time.sleep(10)


if __name__ == "__main__":
    main()
