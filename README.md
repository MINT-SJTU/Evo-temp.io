# Evo-RL

<p align="center">
  <strong>Real-robot reinforcement learning on top of LeRobot.</strong><br/>
  Human-in-the-loop collection, value learning, ACP indicator annotation, and policy training in one practical loop.
</p>

<p align="center">
  <a href="./website/index.html">Project Website</a> •
  <a href="./docs/source">Docs Source</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="./docs/outreach/wechat_push_draft_zh.md">WeChat Draft (ZH)</a>
</p>

<p align="center">
  <img alt="license" src="https://img.shields.io/badge/license-Apache--2.0-2ea44f"/>
  <img alt="python" src="https://img.shields.io/badge/python-3.10%2B-3776AB"/>
  <img alt="status" src="https://img.shields.io/badge/status-active%20development-0A7EA4"/>
  <img alt="robot" src="https://img.shields.io/badge/focus-real--robot%20RL-1f6feb"/>
</p>

## Why Evo-RL

Evo-RL extends LeRobot for real-world RL workflows where pure behavior cloning is not enough.

It focuses on a production-minded closed loop:

1. Collect real-robot trajectories with explicit success/failure and intervention signals.
2. Train a value model from those trajectories.
3. Annotate frame-level value/advantage/indicator back into the dataset.
4. Train ACP-aware policies that consume those signals.
5. Deploy, evaluate, collect new data, and iterate.

## What Changed vs `main`

As of **February 26, 2026**, `main..kye/main` contains:

- `102` files changed
- `+6515 / -3333` lines

Key additions in this branch:

- Human-in-the-loop recording CLI with intervention and episode outcome labeling.
- Dataset quality report CLI (`lerobot-dataset-report`).
- Value training CLI (`lerobot-value-train`) and value inference CLI (`lerobot-value-infer`).
- ACP prompt conditioning integration in `lerobot-train`.
- Dual-arm SO101 teleop support and related tests.

## Core Pipeline

```text
[Real-robot HIL collection]
        |
        v
[Dataset quality check]
        |
        v
[Value training]
        |
        v
[Value/Adv/Indicator annotation]
        |
        v
[ACP-aware policy training]
        |
        v
[Real-robot evaluation and next iteration]
```

## Main Capabilities

| Stage | CLI / Module | Output |
|---|---|---|
| HIL data collection | `lerobot-human-inloop-record` | `episode_success`, `collector_policy_id`, intervention traces |
| Dataset inspection | `lerobot-dataset-report` | schema/quality/length histogram/success-failure ratios |
| Value training | `lerobot-value-train` | value checkpoints |
| Value inference | `lerobot-value-infer` | `complementary_info.value` (+ ACP fields when enabled) |
| Policy training | `lerobot-train` + ACP hook | ACP-conditioned policy fine-tuning |

## Installation

```bash
git clone <your-fork-url> evo-rl
cd evo-rl
conda activate lerobot
pip install -e .
```

Recommended extras:

```bash
# value pipeline dependencies
pip install -e ".[pi]"

# SO100/SO101 hardware stack
pip install -e ".[feetech]"
```

## Quick Start

### 0) Shared vars

```bash
export DATASET_REPO_ID=local/evo_hil_pick_place_v0
export DATASET_ROOT=~/.cache/huggingface/lerobot
```

### 1) HIL recording

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

Optional policy-assisted mode:

```bash
# append to the command above
--policy.path=<policy_repo_or_local_path> \
--acp_inference.enable=true \
--acp_inference.use_cfg=false
```

Default hotkeys:

- `i`: toggle intervention takeover
- `s`: mark success and end episode
- `f`: mark failure and end episode

### 2) Dataset report

```bash
lerobot-dataset-report \
  --dataset ${DATASET_REPO_ID} \
  --root ${DATASET_ROOT}
```

JSON mode:

```bash
lerobot-dataset-report \
  --dataset ${DATASET_REPO_ID} \
  --root ${DATASET_ROOT} \
  --json
```

### 3) Train value model

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

### 4) Infer value/ACP labels and write back to dataset

```bash
lerobot-value-infer \
  --dataset.repo_id=${DATASET_REPO_ID} \
  --dataset.root=${DATASET_ROOT} \
  --inference.checkpoint_path=outputs/value_train/evo_value_demo \
  --inference.checkpoint_ref=last \
  --runtime.batch_size=64 \
  --acp.enable=true
```

Optional overlay video export:

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

## Current Real-World Snapshot (Replaceable)

This section can be replaced by your latest public numbers.

- Platform: dual-arm SO101 towel folding.
- D0 dataset: `300` episodes, `413,134` frames, ~`3.82h` at `30 FPS`.
- Scaling observation (current internal run): Full FT at 100% data reached non-zero success, while several LoRA variants lagged.

For publication, prefer linking a structured experiment table on the project website.

## Repository Layout

Key Evo-RL components:

- `src/lerobot/scripts/lerobot_human_inloop_record.py`
- `src/lerobot/scripts/recording_hil.py`
- `src/lerobot/scripts/recording_loop.py`
- `src/lerobot/scripts/lerobot_dataset_report.py`
- `src/lerobot/scripts/lerobot_value_train.py`
- `src/lerobot/scripts/lerobot_value_infer.py`
- `src/lerobot/values/pistar06/`
- `src/lerobot/rl/acp_hook.py`
- `src/lerobot/rl/acp_dataset_stats.py`

## Project Website

A polished, animated project page is included in [`website/`](./website/):

```bash
cd website
python -m http.server 8000
# open http://localhost:8000
```

All demo/media blocks are placeholders by design so you can drop in your final videos/figures.

## Roadmap

- Public release of dual-arm SO101 training/eval configs.
- Standardized experiment cards (data budget, robot hours, success metrics).
- Better safety tooling for intervention-heavy online RL loops.
- Broader robot coverage beyond SO-series.

## Relationship to Upstream LeRobot

- Evo-RL is a LeRobot-derived branch, not a clean-room rewrite.
- Package/CLI namespace remains `lerobot` and `lerobot-*`.
- Upstream sync/rebase is expected as LeRobot evolves.

## Citation

If Evo-RL is useful in your work, cite this repository and the upstream LeRobot project.

```bibtex
@misc{evorl2026,
  title        = {Evo-RL: Real-Robot Reinforcement Learning on top of LeRobot},
  author       = {Evo-RL Contributors},
  year         = {2026},
  howpublished = {\url{https://github.com/Elvin-yk/evo-lerobot}}
}
```

## License

Apache-2.0. See [LICENSE](./LICENSE).
