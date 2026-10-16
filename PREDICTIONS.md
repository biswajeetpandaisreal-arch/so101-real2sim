# Predictions (written before any real data was recorded or any policy trained)

The git commit that adds this file precedes the commits that add the real
recordings, the identified actuator model and every policy result. Each
prediction states what would count as it being wrong. Results are reported
against this list in the README whether they confirm it or not.

## Digital twin

- **T1 – kinematics.** After the best 2-D rigid alignment (which absorbs where the
  tape frame was laid down), the model's fixed-jaw tip is within **10 mm RMS** of the
  tape-measured crosses. *Wrong if* aligned RMS > 10 mm.
- **T2 – the wrist-roll lag.** The ~5° wrist-roll lag seen during teleoperation is a
  steady-state error of the servo's position loop (LeRobot sets P = 16, half the
  default, plus the servo dead band), not a calibration offset: 1–2° steps of
  wrist_roll produce little or no motion, and the error does not grow with step size.
  *Wrong if* small steps are tracked to < 1° or the error scales with the step.
- **T3 – actuator identification.** Parameters fitted on the excitation run reduce
  the joint tracking RMSE on **held-out teleoperation recordings** for every arm
  joint, compared with the MuJoCo Menagerie defaults. *Wrong if* any arm joint gets
  worse on held-out data.

## Generated trajectories on the real arm (open loop, no policy)

- **R1.** At least **4 of 6** runs succeed at the interpolation spots (I1–I3), and
  extrapolation spots (E1–E3) succeed no more often than interpolation spots.
  *Wrong if* interpolation < 4/6, or extrapolation > interpolation.

## Policies (ACT, 3 training seeds, pooled; checkpoints chosen on validation only)

Conditions: `real` = sim replays of the recorded demonstrations only;
`gen200` = those + 200 generated demonstrations; `naive200` = those + 200 episodes
with the cube moved but the arm's motion unchanged (negative control).

- **P1 – memorisation reproduced.** `real` succeeds on ≥ 50% of seen-spot trials but
  ≤ 20% of extrapolation trials. *Wrong if* `real` extrapolates well (> 20%) — then
  memorisation was not the failure mode here.
- **P2 – generation fixes it.** `gen200` > `real` on both interpolation and
  extrapolation, with Fisher's exact p < 0.05 on extrapolation (pooled over seeds).
  *Wrong if* not significant.
- **P3 – it is the adapted motion, not just moving the cube.** `naive200` does **not**
  beat `real` on interpolation or extrapolation (difference within the 95% CIs or
  worse). *Wrong if* `naive200` ≈ `gen200`: the gain would then come from visual
  diversity alone, not from the trajectory adaptation.
- **P4 – no forgetting.** `gen200` on seen spots is no more than 10 points below `real`.
- **P5.** For every condition, interpolation ≥ extrapolation.
- **F1 – failure mode.** `real`'s extrapolation failures are mostly `no_contact`
  (reaching for a memorised spot), while `gen200`'s are mostly `pushed` /
  `not_delivered` (reaching the right place, imprecise grasp).

## Ablations (1 seed unless stated)

- **A1 – shift timing (generation level).** Shifting before the final descent gives a
  higher physics success rate for generated trajectories than shifting during it
  (scripted pilot: 30/30 vs 18/30).
- **A2 – amount of generated data.** Extrapolation success rises with 25 → 100 → 200
  generated demonstrations.
- **A3 – physics filter.** Training on unfiltered generated data (failures included)
  is worse on extrapolation than filtered `gen200`.
