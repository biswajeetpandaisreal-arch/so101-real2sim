"""README media.

  results/media/real_vs_sim.gif      one recorded demo: real side + wrist cameras next
                                     to the simulated replay of the same commands
  results/media/generated.gif        trajectories generated from that demo for the
                                     interpolation and extrapolation marks and random
                                     positions, as they run in physics

    python scripts/make_media.py --episode 13
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from real2sim.generate import generate  # noqa: E402
from real2sim.kinematics import SO101IK  # noqa: E402
from real2sim.layout import read_layout  # noqa: E402
from real2sim.sim_env import SO101Env  # noqa: E402
from scripts.replay import load_demos, read_spots  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "results" / "media"
STRIDE = 3                                       # 30 Hz -> 10 fps GIFs


def label(img, text):
    from PIL import ImageDraw
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 8 + 7 * len(text), 16], fill=(0, 0, 0))
    d.text((4, 2), text, fill=(255, 255, 255))
    return np.asarray(im)


def real_frames(data, repo_id, episode, start, end):
    import torch  # noqa: F401  (LeRobotDataset returns tensors)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(repo_id, root=data, video_backend="pyav")
    idx = np.where(np.asarray(ds.hf_dataset["episode_index"]) == episode)[0]
    out = []
    for i in idx[start:end:STRIDE]:
        item = ds[int(i)]
        out.append([(item[f"observation.images.{c}"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                     for c in ("side", "wrist")])
    return out


def save_gif(frames, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    ims = [Image.fromarray(f) for f in frames]
    ims[0].save(path, save_all=True, append_images=ims[1:], duration=100, loop=0, optimize=True)
    print(f"wrote {path} ({len(ims)} frames)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/so101_real")
    ap.add_argument("--repo-id", default="local/so101_real")
    ap.add_argument("--episode", type=int, default=13)
    a = ap.parse_args()
    data = Path(a.data)
    spots = read_spots(data / "episodes.csv")
    ep = next(e for e in load_demos(data, a.repo_id, spots) if e["episode"] == a.episode)
    sp = spots[a.episode]
    start, end = sp["frames"]
    env = SO101Env(render_size=(240, 320))

    # real vs sim, same commands
    real = real_frames(data, a.repo_id, a.episode, start, end)
    env.reset(sp["xy"], sp["yaw"], q0_lerobot=ep["state"][0])
    sim = []
    for t, act in enumerate(ep["action"]):
        if t % STRIDE == 0:
            sim.append([env.render("overview", pretty=True), env.render("wrist_cam", pretty=True)])
        env.step(act)
    n = min(len(real), len(sim))
    frames = []
    for (rs, rw), (ss, sw) in zip(real[:n], sim[:n]):
        rs = np.asarray(Image.fromarray(rs).resize((320, 240)))
        rw = np.asarray(Image.fromarray(rw).resize((320, 240)))
        top = np.hstack([label(rs, "real: side camera"), label(ss, "sim: replay of the same commands")])
        bot = np.hstack([label(rw, "real: wrist camera"), label(sw, "sim: wrist camera")])
        frames.append(np.vstack([top, bot]))
    save_gif(frames, OUT / "real_vs_sim.gif")

    # generated demonstrations from this one recording
    ik = SO101IK(env)
    layout = read_layout()
    rng = np.random.default_rng(3)
    targets = [("mark 1 (interp.)", layout["1"]["xy"]), ("mark 6 (extrap.)", layout["6"]["xy"])]
    targets += [(f"random {k + 1}", np.array([rng.uniform(0.22, 0.38), rng.uniform(-0.08, 0.10)])) for k in range(2)]
    clips = []
    for name, xy in targets:
        acts = generate(env, ik, ep["action"], sp["xy"], sp["yaw"], xy, 0.0)
        env.reset(xy, 0.0, q0_lerobot=acts[0])
        clip = []
        for t, act in enumerate(acts):
            if t % STRIDE == 0:
                clip.append(label(env.render("overview", pretty=True), name))
            env.step(act)
        clip.append(label(env.render("overview", pretty=True), f"{name}: {'in the box' if env.success() else 'failed'}"))
        clips.append(clip)
    n = max(len(c) for c in clips)
    clips = [c + [c[-1]] * (n - len(c)) for c in clips]
    grid = [np.vstack([np.hstack([clips[0][t], clips[1][t]]), np.hstack([clips[2][t], clips[3][t]])])
            for t in range(n)]
    save_gif(grid, OUT / "generated.gif")


if __name__ == "__main__":
    main()
