# Teaching a robot to pick where nobody showed it: real SO-101 demos → measured digital twin → generated data

I recorded my own pick-and-place demonstrations on a real SO-101 arm, built and
**measured** a MuJoCo digital twin of the setup, used the twin to turn 12 demos into
hundreds of position-diverse ones, and tested whether that stops an imitation-learning
policy from memorising object positions, which is the failure that sank the VLA in my MSc thesis
(0/216 successes at unseen positions).

![real vs sim](results/media/real_vs_sim.gif)

*One real demonstration (left) and the twin replaying the same leader commands
through identified actuator models (right).*

<!-- TLDR -->

---

## 1. The question

Imitation-learning policies trained on a handful of demonstrations tend to learn
*where the object was* rather than *how to pick it*. In my thesis a fine-tuned VLA
scored well at the demonstrated positions and 0/216 elsewhere. Collecting demonstrations
everywhere on the table does not scale, so this project asks:

> Can a small set of real demonstrations, adapted to new object positions inside a
> digital twin that has been *measured against the real arm*, teach a policy to pick at
> positions nobody demonstrated?
> And is it the *adapted motion* that helps, or just showing the object in more places?

Every prediction was written down **before** any data was recorded or any policy trained:
[PREDICTIONS.md](PREDICTIONS.md) (commit `0613d79` precedes all data and results commits).
Section 6 scores each one, including the ones I got wrong.

## 2. Setup

* **Robot:** SO-101 leader + follower (Feetech STS3215 servos), LeRobot 0.6.1 on Windows 11.
* **Task:** pick an LED tealight (cylinder, 38 mm × 23 mm) and drop it in a box.
* **Table:** six taped marks. Demonstrations were recorded at marks **2, 3, 4, 5** only. Mark **1**
  lies inside them (interpolation test) and mark **6** lies beyond them (extrapolation test).
* **Data:** 20 episodes, `lerobot-record`, 30 Hz joint commands (leader) and positions
  (follower), wrist camera and side camera. 16 contain a grasp; the rest were idle or
  cut short (operator error, listed below).
* **Sim:** MuJoCo, SO-101 model from MuJoCo Menagerie, with the scene rebuilt from
  measurements of the real one.

## 3. Measuring the twin before trusting it

| what | how | result |
|---|---|---|
| where the objects are | forward kinematics at the moment the real jaws stall on the tealight, per demo | marks recovered to within 0.5–2.2 cm across demos (hand placement) |
| independent check | top-down photo of the table, similarity-fitted to the arm's measurements | ~1 cm residual on 5 marks |
| joint limits | calibrated range of the real follower | real arm rests *outside* the Menagerie limits (wrist_flex −102° vs ±95°); limits widened |
| actuators | 200 Hz excitation (steps, small steps, sines) per joint; least squares on kp, damping, friction, armature | fitted stiffness 16–62 vs Menagerie 998; held-out error down on every joint (table below) |
| end to end | replay recorded leader commands in the twin | 9/16 demos succeed with Menagerie actuators → **13/16 with identified actuators** |

The last row matters most. The real STS3215 servos run LeRobot's soft P gain and lag the
operator by up to 40° on fast moves. The default model tracks commands almost perfectly, so
it whipped the object out of the jaws. With identified actuators the twin reproduces the
real outcome from the *same commands* the real arm received.

<!-- TWIN -->

## 4. Method: object-centric demonstration generation

A simplified [MimicGen](https://mimicgen.github.io) adapted to a 5-DoF arm and to *real*
recordings (`real2sim/generate.py`):

1. Segment the demo at the grasp and release (gripper command; real demos start and end
   with the gripper shut, so the grasp is the first close *after* an opening).
2. Compute the grasp-centre path and approach direction with forward kinematics.
3. Shift the path by the object's displacement. The shift is **ramped in while the gripper is still high, finished
   before the final descent**, held through the grasp and ramped out before the release (the
   box does not move).
4. Solve inverse kinematics along the new path. The SO-101 cannot point straight down
   everywhere, so position is exact and the approach direction is a soft goal. Gripper timing is kept.
5. Run every generated trajectory in physics and **keep only successes**.

Generation sources are the 12 recorded demos that the twin reproduces. The negative control
(`naive`) moves the object but leaves the arm's motion unchanged.

![generated](results/media/generated.gif)

*Trajectories generated from one real demonstration, for the interpolation mark, the
extrapolation mark and two random positions.*

## 5. Results

<!-- RESULTS -->

## 6. Predictions vs outcome

<!-- PREDICTIONS -->

## 7. Things that went wrong, and what they taught me

* **The twin was "right" and still wrong.** The kinematics matched the real arm to about 1 cm on the first
  try, but replaying leader commands failed 7 times in 16. The gap was dynamics: real servos lag and smooth,
  the model does not. Measuring the actuators fixed it, so "the twin looks right" is not evidence.
* **A static dead band was the wrong explanation (prediction T2).** I attributed a 5° wrist-roll lag to the
  servo dead band. The servo registers say the dead zone is one encoder tick, and step tests show ≤0.4°
  steady-state error. The lag is dynamic: a soft position loop plus a 10°/frame safety clamp.
* **Data collection bugs are data bugs.** LeRobot's recorder listens to keys system-wide and maps `n`/`r`/`q`
  to next/re-record/quit. Typing a chat message ended or restarted demonstrations, producing 2-second and
  truncated episodes. The recorder here only listens to the arrow keys. Demos are trimmed to exactly one
  pick using the grasp and the return to rest, and only episodes the twin reproduces are used as sources.
* **The grasp detector assumed scripted data.** My generator found the grasp as "first frame with the gripper
  closed". Real demonstrations start with the gripper closed at rest, so no plan succeeded until that was fixed.
* **Rendering, not learning, was the bottleneck.** MuJoCo was rendering 400k-triangle CAD meshes with shadows on
  the laptop's integrated GPU, and LeRobot wrote every frame as a PNG before encoding. Collision geometry, no
  shadows and streaming encoding made dataset writing about 100× faster. Policies are trained and tested on that
  same rendering.

## 8. Limitations

* Policies are trained and evaluated **in simulation**, on simulated images. The real
  data supplies the motions, not the pixels. Deploying a learned policy on the real arm
  needs a sim-to-real step for vision (co-training on the recorded camera data is the obvious next step,
  and the real wrist and side videos are in the dataset for that).
* The generated trajectories *are* tested on the real arm, open loop, without a policy (section 5).
* One operator, one object, one box position, 12 source demos.
* The physics filter and the policy evaluation use the same simulator, so generated data is
  "validated" by the judge. The real-arm execution test is the check on that.
* Training is compute-matched (8000 steps, batch 16); ablations have one seed.
* The positions of mark 6 and the box come from the photo fit (about 1 cm), not from a grasp.

## 9. Reproduce

```bash
python -m venv .venv && .venv\Scripts\pip install "lerobot[feetech,dataset,training]==0.6.1" mujoco scipy
python scripts/make_episode_list.py                 # object positions + task frames from the recordings
python scripts/sysid_fit.py --excite data/sysid/excite_*.npz --validate data/so101_real
python scripts/check_demos.py                       # which demos the twin reproduces
python scripts/run_experiments.py --parallel 3      # build-free if data/datasets exists; trains + evaluates all runs
python scripts/analyse.py                           # every number and figure in this README
```

Recording your own: `scripts/record_real.py` (two cameras, arrows-only keys),
`scripts/sysid_excite.py` (servo excitation), `scripts/execute_real.py` (generated
trajectories on the real arm).

## 10. Related work

MimicGen (Mandlekar et al., CoRL 2023) generates large datasets from a few demonstrations
in simulation; this is a simplified version for a 5-DoF arm, driven by real recordings.
DemoGen (2025) does the same adaptation on real point clouds; Real2Render2Real (CoRL 2025)
renders robot data from a phone scan and a human video. so101-nexus provides SO-101 MuJoCo
environments without real-to-sim replay or generation.
