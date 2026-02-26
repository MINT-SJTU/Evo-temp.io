# Evo-RL

<p align="center">
  <strong>Open-source real-world RL toolkit for LeRobot101 (SO101), with continuous model, algorithm, and dataset releases.</strong><br/>
  LeRobot101 is live now. AgileX PiPER robot-arm support is coming soon.
</p>

<p align="center">
  <a href="./website/index.html">Project Website</a> •
  <a href="#reproduce-current-release">Reproduce Current Release</a> •
  <a href="#community-program">Community Program</a> •
  <a href="./docs/outreach/wechat_push_draft_zh.md">WeChat Draft (ZH)</a>
</p>

<p align="center">
  <img alt="license" src="https://img.shields.io/badge/license-Apache--2.0-2ea44f"/>
  <img alt="python" src="https://img.shields.io/badge/python-3.10%2B-3776AB"/>
  <img alt="platform" src="https://img.shields.io/badge/platform-LeRobot101%20(SO101)-1f6feb"/>
  <img alt="status" src="https://img.shields.io/badge/AgileX%20PiPER-coming%20soon-ff8f5b"/>
</p>

## Positioning

Evo-RL is a **continuous open-source program**, not a one-time demo release.

- Current platform: **LeRobot101 / SO101**.
- Next platform: **AgileX PiPER robot arms** (coming soon, https://www.agilex.ai/).
- Goal: build a community where tasks on both platforms can be uploaded, reproduced, and benchmarked with shared protocols.

## Core Selling Points

1. To our knowledge (within current open-source LeRobot ecosystem), Evo-RL is the first project continuously pushing real-world RL releases on LeRobot101.
2. End-to-end loop already implemented: data collection -> value -> indicator -> policy -> real-world re-collection.
3. Engineering-first design: intervention/outcome labels, dataset quality report, indicator-conditioned training integration, and reproducible CLI chain.
4. Community-first roadmap: open task uploads and cross-platform benchmark evolution.

## What Is Already Implemented

As of commit `852b23cb` (2026-02-26), compared to `main`, core RL tooling includes:

- `101` core files changed (`website` and outreach files excluded)
- `+6336 / -3224` lines
- New CLIs:
  - `lerobot-human-inloop-record`
  - `lerobot-dataset-report`
  - `lerobot-value-train`
  - `lerobot-value-infer`
- Indicator-conditioned policy path integrated into `lerobot-train`
- Value/advantage/indicator write-back pipeline for iterative real-world training

Recompute stats with:

```bash
git diff --shortstat main..kye/main -- . \
  ':(exclude)website' \
  ':(exclude)docs/outreach' \
  ':(exclude).github/workflows/deploy-website-pages.yml' \
  ':(exclude)README.md'
```

## Core Pipeline

```text
[HIL collection on real robot]
         |
         v
[Dataset quality report]
         |
         v
[Value training]
         |
         v
[Value / Advantage / Indicator annotation]
         |
         v
[Indicator-conditioned policy training]
         |
         v
[Real-world rollout + next iteration]
```

## Reproduce Current Release

### Tested environment

- Ubuntu 22.04
- Python 3.10
- CUDA 12.x class environment
- SO-series leader/follower arm + USB camera (640x480@30)

### 1) Install

```bash
git clone https://github.com/Elvin-yk/evo-lerobot.git
cd evo-lerobot
conda activate lerobot
pip install -e .
pip install -e ".[pi]"      # value pipeline deps
pip install -e ".[feetech]" # SO101 related deps
```

### 2) Prepare variables

```bash
export DATASET_REPO_ID=<your_dataset_repo_id>
export DATASET_ROOT=~/.cache/huggingface/lerobot
```

### 3) Hardware preflight (highly recommended)

```bash
lerobot-find-port
lerobot-find-cameras
```

Confirm:

- Correct follower/leader serial ports
- Correct camera index/path
- Current user has serial device permission

### 4) Human-in-loop recording

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

Hotkeys:

- `i`: toggle intervention
- `s`: mark success and end episode
- `f`: mark failure and end episode

### 5) Dataset quality report

```bash
lerobot-dataset-report --dataset ${DATASET_REPO_ID} --root ${DATASET_ROOT}
lerobot-dataset-report --dataset ${DATASET_REPO_ID} --root ${DATASET_ROOT} --json
```

Expected output:

- Terminal report with success/failure/intervention metrics
- JSON report for logging and experiment cards

### 6) Value training

```bash
lerobot-value-train \
  --dataset.repo_id=${DATASET_REPO_ID} \
  --dataset.root=${DATASET_ROOT} \
  --dataset.download_videos=true \
  --value.type=pistar06 \
  --batch_size=16 \
  --steps=2000 \
  --output_dir=outputs/value_train/evo_value_demo \
  --job_name=evo_value_demo
```

Expected output:

- Value checkpoints under `outputs/value_train/evo_value_demo`

### 7) Value infer + advantage-indicator labels

```bash
lerobot-value-infer \
  --dataset.repo_id=${DATASET_REPO_ID} \
  --dataset.root=${DATASET_ROOT} \
  --inference.checkpoint_path=outputs/value_train/evo_value_demo \
  --inference.checkpoint_ref=last \
  --runtime.batch_size=64 \
  --acp.enable=true
```

Expected output:

- Dataset fields written/updated:
  - `complementary_info.value`
  - `complementary_info.advantage`
  - `complementary_info.acp_indicator`

Optional visualization export:

```bash
lerobot-value-infer \
  --dataset.repo_id=${DATASET_REPO_ID} \
  --dataset.root=${DATASET_ROOT} \
  --inference.checkpoint_path=outputs/value_train/evo_value_demo \
  --inference.checkpoint_ref=last \
  --acp.enable=true \
  --viz.enable=true \
  --viz.episodes=all \
  --viz.output_dir=outputs/value_infer/viz
```

### 8) Indicator-conditioned policy training

CLI flags use `--acp.*`, where `acp_indicator` is the binary advantage-indicator field.

```bash
lerobot-train \
  --dataset.repo_id=${DATASET_REPO_ID} \
  --dataset.root=${DATASET_ROOT} \
  --policy.type=pi05 \
  --policy.pretrained_path=lerobot/pi05_base \
  --acp.enable=true \
  --acp.indicator_field=complementary_info.acp_indicator \
  --steps=3000
```

## Current Evidence Snapshot

- Platform: dual-arm SO101 task iteration
- D0 dataset scale: `300` episodes, `413,134` frames, ~`3.82h` at `30 FPS`
- Internal observed trend: `100% data + Full FT` forms a viable baseline in current setup

More benchmark cards and protocol details are published on the project website as releases progress.

## Community Program

We are opening a community track for tasks on:

- LeRobot101 (SO101)
- AgileX PiPER robot arms (coming soon, https://www.agilex.ai/)

Current submission flow:

1. Open an Issue with task setup and success definition.
2. Share dataset link + exact training/inference commands.
3. Submit result table + short video for benchmark integration.

## Website

Project page is in [`website/`](./website/):

```bash
cd website
python -m http.server 8000
```

## Relationship to Upstream LeRobot

- This repository is LeRobot-derived and keeps the `lerobot-*` CLI namespace.
- Evo-RL extends upstream with real-world RL loop tooling and continuous release workflow.

## Citation

```bibtex
@misc{evorl2026,
  title        = {Evo-RL: Continuous Open-Source Real-World RL on LeRobot101 and Beyond},
  author       = {Evo-RL Contributors},
  year         = {2026},
  howpublished = {\url{https://github.com/Elvin-yk/evo-lerobot}}
}
```

## License

Apache-2.0. See [LICENSE](./LICENSE).
