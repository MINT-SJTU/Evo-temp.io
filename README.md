# Evo-RL

Evo-RL is a real-robot reinforcement-learning extension built on top of LeRobot.
It keeps the LeRobot hardware/dataset/policy foundation and adds a practical pipeline for:

- human-in-the-loop (HIL) data collection with intervention labels,
- dataset quality reporting,
- value-model training and offline inference for ACP-style frame annotation.

This README is for researchers using this repository for the first time.

## Project Scope

Evo-RL focuses on closing the loop from real-world data collection to training-time conditioning:

1. Collect HIL trajectories on real robots, including episode-level success/failure labels.
2. Inspect dataset quality and intervention statistics before training.
3. Train a value model from recorded trajectories.
4. Infer per-frame value/advantage/indicator labels and write them back into the dataset.
5. Reuse those labels in `lerobot-train` with ACP prompt conditioning.

## Main Capabilities

- `lerobot-human-inloop-record`
  - policy + teleop mirrored execution support
  - intervention state machine with keyboard toggles
  - per-episode success/failure labels (`episode_success`)
  - frame-level provenance and intervention fields:
    `complementary_info.policy_action`,
    `complementary_info.is_intervention`,
    `complementary_info.state`,
    `complementary_info.collector_policy_id`
- `lerobot-dataset-report`
  - dataset schema and task inventory
  - declared vs actual episode/frame counts
  - success/failure and intervention ratios
  - 20-bin episode length histogram
- `lerobot-value-train`
  - end-to-end value training pipeline from LeRobot dataset
  - checkpoint save and optional hub upload
- `lerobot-value-infer`
  - loads value checkpoints and computes per-frame:
    `complementary_info.value`,
    `complementary_info.advantage`,
    `complementary_info.acp_indicator`
  - writes annotations in-place into dataset parquet/meta files
  - optional push of annotated dataset to hub
- ACP-aware policy training in `lerobot-train`
  - optional prompt hook controlled by `acp.*` config
  - reads binary indicator field and appends positive/negative tags to `task`

## Installation

### 1) Clone and setup environment

```bash
git clone <your-fork-url> evo-rl
cd evo-rl
conda activate lerobot
```

If you create a fresh env, use Python 3.10+.

### 2) Install package

Minimal install:

```bash
pip install -e .
```

Recommended for value pipeline (`transformers` needed):

```bash
pip install -e ".[pi]"
```

Note: current value-module runtime errors may still suggest `lerobot[pi0]`; in this branch use `lerobot[pi]`.

For SO100/SO101 teleoperation hardware:

```bash
pip install -e ".[feetech]"
```

## Quick Start

The commands below use existing CLI entrypoints and current argument names from this branch.

### 0) Shared variables

```bash
export DATASET_REPO_ID=local/evo_hil_pick_place_v0
export DATASET_ROOT=~/.cache/huggingface/lerobot
```

### 1) Human-in-loop recording (minimum runnable template)

```bash
lerobot-human-inloop-record \
  --robot.type=so100_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
  --teleop.type=so100_leader \
  --teleop.port=/dev/ttyACM1 \
  --dataset.repo_id=${DATASET_REPO_ID} \
  --dataset.root=${DATASET_ROOT} \
  --dataset.single_task="Pick and place the red block" \
  --dataset.num_episodes=20 \
  --dataset.push_to_hub=false
```

Optional policy-assisted HIL (append these flags to the command above):

```bash
  --policy.path=<policy_repo_or_local_path> \
  --acp_inference.enable=true \
  --acp_inference.use_cfg=false
```

Default hotkeys during recording:

- `i`: toggle intervention takeover
- `s`: mark episode `success` and end
- `f`: mark episode `failure` and end

### 2) Dataset report

```bash
lerobot-dataset-report \
  --dataset ${DATASET_REPO_ID} \
  --root ${DATASET_ROOT}
```

JSON output:

```bash
lerobot-dataset-report \
  --dataset ${DATASET_REPO_ID} \
  --root ${DATASET_ROOT} \
  --json
```

### 3) Value training

```bash
lerobot-value-train \
  --dataset.repo_id=${DATASET_REPO_ID} \
  --dataset.root=${DATASET_ROOT} \
  --dataset.download_videos=true \
  --train.max_steps=2000 \
  --train.batch_size=16 \
  --output_dir=outputs/evo_value_demo
```

### 4) Value inference (write annotations into dataset)

```bash
lerobot-value-infer \
  --dataset.repo_id=${DATASET_REPO_ID} \
  --dataset.root=${DATASET_ROOT} \
  --inference.checkpoint_root=outputs/evo_value_demo/value/checkpoints \
  --inference.checkpoint_ref=last \
  --runtime.batch_size=64 \
  --push_to_hub=false
```

After inference, dataset frames contain:

- `complementary_info.value`
- `complementary_info.advantage`
- `complementary_info.acp_indicator`

Optional overlay video export:

```bash
lerobot-value-episode-viz \
  --repo-id ${DATASET_REPO_ID} \
  --root ${DATASET_ROOT} \
  --episodes all \
  --output-dir outputs/value_vis
```

`lerobot-value-episode-viz` requires the three annotation fields above to exist.
If they are missing, run `lerobot-value-infer` first.

## Repository Layout

Key Evo-RL additions are concentrated in:

- `src/lerobot/scripts/lerobot_human_inloop_record.py`
- `src/lerobot/scripts/recording_hil.py`
- `src/lerobot/scripts/recording_loop.py`
- `src/lerobot/scripts/lerobot_dataset_report.py`
- `src/lerobot/scripts/lerobot_value_train.py`
- `src/lerobot/scripts/lerobot_value_infer.py`
- `src/lerobot/scripts/lerobot_value_episode_viz.py`
- `src/lerobot/value/` (value model/config/io/preprocess/telemetry)
- `src/lerobot/rl/acp_hook.py`
- `src/lerobot/rl/acp_dataset_stats.py`

Core LeRobot base remains in:

- `src/lerobot/robots/`
- `src/lerobot/teleoperators/`
- `src/lerobot/datasets/`
- `src/lerobot/policies/`
- `src/lerobot/scripts/lerobot_train.py`

## Relationship to Upstream LeRobot

- This repository is a LeRobot-derived branch, not a clean-room rewrite.
- Package name and CLI namespace remain `lerobot` / `lerobot-*`.
- Existing LeRobot workflows still apply; Evo-RL extends them with HIL annotation and value/ACP tooling.
- Upstream sync is expected; behavior can change after future rebases/merges.

## Known Limitations

- Value pipeline currently supports `SiglipGemmaValueConfig` only.
- Value preprocessing/model loading depends on `transformers`; missing dependency causes runtime import errors.
- Value inference updates dataset parquet/meta files in place. Back up datasets before large runs.
- `lerobot-human-inloop-record` requires teleop config; ACP inference options require a policy.
- Headless environments may not support keyboard listener hotkeys.
- ACP hook expects binary integer indicators (`0/1`) in the configured indicator field.
- Resuming old policy+teleop datasets (created before `complementary_info.*` fields were introduced) can fail metadata compatibility checks.

## License

Apache-2.0. See `LICENSE`.
