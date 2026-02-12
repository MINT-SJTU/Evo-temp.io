#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.configs.value_train import ValueTrainPipelineConfig
from lerobot.value.algorithms import EpisodeTargetInfo


def array_stats(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return {
            "count": 0,
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "q01": 0.0,
            "q10": 0.0,
            "q50": 0.0,
            "q90": 0.0,
            "q99": 0.0,
        }
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "q01": float(np.quantile(arr, 0.01)),
        "q10": float(np.quantile(arr, 0.10)),
        "q50": float(np.quantile(arr, 0.50)),
        "q90": float(np.quantile(arr, 0.90)),
        "q99": float(np.quantile(arr, 0.99)),
    }


def stats_per_task(values: np.ndarray, task_indices: np.ndarray) -> dict[int, dict[str, float]]:
    result: dict[int, dict[str, float]] = {}
    for task_idx in np.unique(task_indices):
        mask = task_indices == task_idx
        result[int(task_idx)] = array_stats(values[mask])
    return result


def log_array_stats(name: str, values: np.ndarray) -> dict[str, float]:
    stats = array_stats(values)
    logging.info(
        "%s stats | count=%d min=%.6f max=%.6f mean=%.6f std=%.6f q10=%.6f q50=%.6f q90=%.6f",
        name,
        stats["count"],
        stats["min"],
        stats["max"],
        stats["mean"],
        stats["std"],
        stats["q10"],
        stats["q50"],
        stats["q90"],
    )
    return stats


def log_pipeline_summary(cfg: ValueTrainPipelineConfig) -> None:
    logging.info(
        "Value pipeline summary | c_fail_coef=%.4f n_step=%d positive_ratio=%.4f "
        "value_field=%s advantage_field=%s indicator_field=%s",
        cfg.targets.c_fail_coef,
        cfg.acp.n_step,
        cfg.acp.positive_ratio,
        cfg.acp.value_field,
        cfg.acp.advantage_field,
        cfg.acp.indicator_field,
    )


def log_input_field_check(
    cfg: ValueTrainPipelineConfig,
    state_feature_exists: bool,
    task_feature_exists: bool,
    success_field_exists: bool,
    intervention_field_exists: bool,
) -> None:
    logging.info(
        "Input field check | state_feature=%s(%s) task_feature=%s(%s) success_field=%s(%s) intervention_field=%s(%s)",
        cfg.value.state_feature,
        state_feature_exists,
        cfg.value.task_index_feature,
        task_feature_exists,
        cfg.dataset.success_field,
        success_field_exists,
        cfg.acp.intervention_field,
        intervention_field_exists,
    )


def log_dataset_overview(frame_count: int, num_episodes: int, num_tasks: int, state_dim: int) -> None:
    logging.info(
        "Dataset overview | num_frames=%d num_episodes=%d num_tasks=%d state_dim=%d",
        frame_count,
        num_episodes,
        num_tasks,
        state_dim,
    )


def summarize_task_episodes(
    episode_info: dict[int, EpisodeTargetInfo], task_max_length: dict[int, int]
) -> dict[int, dict[str, float | int]]:
    per_task_episode: dict[int, dict[str, float | int]] = {}
    for task_idx in sorted(task_max_length):
        task_episodes = [ep for ep in episode_info.values() if ep.task_index == task_idx]
        success_count = sum(1 for ep in task_episodes if ep.success)
        total_count = len(task_episodes)
        success_rate = float(success_count / total_count) if total_count > 0 else 0.0
        per_task_episode[task_idx] = {
            "episode_count": total_count,
            "success_rate": success_rate,
            "l_max": int(task_max_length[task_idx]),
        }
        logging.info(
            "Task[%d] episodes=%d success_rate=%.4f l_max=%d",
            task_idx,
            total_count,
            success_rate,
            task_max_length[task_idx],
        )
    return per_task_episode


def log_task_target_reward_stats(
    per_task_target_stats: dict[int, dict[str, float]],
    per_task_reward_stats: dict[int, dict[str, float]],
) -> None:
    for task_idx in sorted(per_task_target_stats):
        t_stats = per_task_target_stats[task_idx]
        r_stats = per_task_reward_stats[task_idx]
        logging.info(
            "Task[%d] target mean=%.6f p50=%.6f | reward mean=%.6f p50=%.6f",
            task_idx,
            t_stats["mean"],
            t_stats["q50"],
            r_stats["mean"],
            r_stats["q50"],
        )


def log_checkpoint_saved(step: int, checkpoint_root: Path) -> None:
    logging.info("Checkpoint saved | step=%d dir=%s", step, checkpoint_root)


def log_inference_outputs(predicted_values: np.ndarray) -> tuple[dict[str, float], list[float]]:
    inference_stats = log_array_stats("Inference value", predicted_values)
    sample_first_5 = predicted_values[:5].astype(float).tolist()
    logging.info(
        "Inference sample | first5=%s",
        np.array2string(predicted_values[:5], precision=6, separator=", "),
    )
    return inference_stats, sample_first_5


def log_advantage_outputs(
    advantages: np.ndarray,
    task_indices: np.ndarray,
    thresholds: dict[int, float],
    indicators: np.ndarray,
) -> tuple[dict[str, float], dict[int, dict[str, float]], dict[int, float]]:
    advantage_stats = log_array_stats("Advantage", advantages)
    per_task_adv_stats = stats_per_task(advantages, task_indices)
    indicator_positive_ratio_per_task: dict[int, float] = {}
    for task_idx in sorted(per_task_adv_stats):
        task_mask = task_indices == task_idx
        task_pos_ratio = float(np.mean(indicators[task_mask].astype(np.float32)))
        indicator_positive_ratio_per_task[task_idx] = task_pos_ratio
        th = thresholds.get(task_idx)
        logging.info(
            "Task[%d] advantage mean=%.6f p50=%.6f threshold=%.6f positive_ratio=%.6f",
            task_idx,
            per_task_adv_stats[task_idx]["mean"],
            per_task_adv_stats[task_idx]["q50"],
            float(th if th is not None else 0.0),
            task_pos_ratio,
        )
    return advantage_stats, per_task_adv_stats, indicator_positive_ratio_per_task


def log_write_modes(cfg: ValueTrainPipelineConfig, write_modes: dict[str, str]) -> None:
    logging.info(
        "Write mode | %s=%s, %s=%s, %s=%s",
        cfg.acp.value_field,
        write_modes.get(cfg.acp.value_field, "unknown"),
        cfg.acp.advantage_field,
        write_modes.get(cfg.acp.advantage_field, "unknown"),
        cfg.acp.indicator_field,
        write_modes.get(cfg.acp.indicator_field, "unknown"),
    )


def save_diagnostics(diagnostics_dir: Path, diagnostics: dict[str, Any]) -> Path:
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    summary_path = diagnostics_dir / "summary.json"
    summary_path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True), encoding="utf-8")
    return summary_path


def build_diagnostics_payload(
    *,
    cfg: ValueTrainPipelineConfig,
    state_feature_exists: bool,
    task_feature_exists: bool,
    success_field_exists: bool,
    intervention_field_exists: bool,
    frame_count: int,
    episode_count: int,
    num_tasks: int,
    state_dim: int,
    per_task_episode: dict[int, dict[str, float | int]],
    target_stats: dict[str, float],
    reward_stats: dict[str, float],
    per_task_target_stats: dict[int, dict[str, float]],
    per_task_reward_stats: dict[int, dict[str, float]],
    last_loss: float,
    last_value_mae: float,
    inference_stats: dict[str, float],
    inference_sample_first_5: list[float],
    advantage_stats: dict[str, float],
    per_task_adv_stats: dict[int, dict[str, float]],
    thresholds: dict[int, float],
    indicator_positive_ratio_global: float,
    indicator_positive_ratio_per_task: dict[int, float],
    write_modes: dict[str, str],
) -> dict[str, Any]:
    return {
        "config_summary": {
            "dataset_repo_id": cfg.dataset.repo_id,
            "state_feature": cfg.value.state_feature,
            "task_feature": cfg.value.task_index_feature,
            "success_field": cfg.dataset.success_field,
            "intervention_field": cfg.acp.intervention_field,
            "c_fail_coef": cfg.targets.c_fail_coef,
            "n_step": cfg.acp.n_step,
            "positive_ratio": cfg.acp.positive_ratio,
        },
        "input_features_presence": {
            cfg.value.state_feature: state_feature_exists,
            cfg.value.task_index_feature: task_feature_exists,
            cfg.dataset.success_field: success_field_exists,
            cfg.acp.intervention_field: intervention_field_exists,
        },
        "dataset_overview": {
            "num_frames": int(frame_count),
            "num_episodes": int(episode_count),
            "num_tasks": int(num_tasks),
            "state_dim": int(state_dim),
            "task_episode_summary": per_task_episode,
        },
        "target_stats": {
            "global": target_stats,
            "per_task": per_task_target_stats,
        },
        "reward_stats": {
            "global": reward_stats,
            "per_task": per_task_reward_stats,
        },
        "training_summary": {
            "last_loss": float(last_loss),
            "last_value_mae": float(last_value_mae),
            "max_steps": int(cfg.train.max_steps),
        },
        "inference_stats": {
            "global": inference_stats,
            "sample_first_5": inference_sample_first_5,
        },
        "advantage_stats": {
            "global": advantage_stats,
            "per_task": per_task_adv_stats,
            "thresholds": {int(k): float(v) for k, v in thresholds.items()},
            "indicator_positive_ratio_global": float(indicator_positive_ratio_global),
            "indicator_positive_ratio_per_task": {
                int(k): float(v) for k, v in indicator_positive_ratio_per_task.items()
            },
        },
        "writeback": {
            "fields": write_modes,
        },
    }
