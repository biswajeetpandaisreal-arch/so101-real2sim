# Record real SO-101 teleoperation demos with lerobot-record (Windows).
#   .\scripts\record_real.ps1 -Episodes 1            # first batch (new dataset)
#   .\scripts\record_real.ps1 -Episodes 6 -Resume    # add episodes to it
# Keys while recording: Right arrow = episode done / skip reset,
#                       Left arrow = redo this episode, Esc = stop.
param([int]$Episodes = 6, [switch]$Resume)

$env:PYTHONIOENCODING = 'utf-8'   # lerobot prints non-ASCII characters
Set-Location (Split-Path $PSScriptRoot -Parent)
.\.venv\Scripts\lerobot-record.exe `
    --robot.type=so101_follower --robot.port=COM3 --robot.id=so101_follower `
    --robot.calibration_dir=config/calibration --robot.max_relative_target=10 `
    --teleop.type=so101_leader --teleop.port=COM4 --teleop.id=so101_leader `
    --teleop.calibration_dir=config/calibration `
    --dataset.repo_id=local/so101_real --dataset.root=data/so101_real `
    --dataset.single_task="Pick up the red cube and place it in the blue basket." `
    --dataset.num_episodes=$Episodes --dataset.fps=30 `
    --dataset.episode_time_s=25 --dataset.reset_time_s=20 `
    --dataset.push_to_hub=false --play_sounds=false `
    --resume=$($Resume.IsPresent.ToString().ToLower())
