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

import logging
import time
from pathlib import Path
from pprint import pformat
from typing import Any

import draccus
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from lerobot.configs.value_train import ValueTrainPipelineConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import cycle
from lerobot.utils.random_utils import set_seed
from lerobot.utils.recording_annotations import EPISODE_SUCCESS, resolve_episode_success_label
from lerobot.utils.utils import auto_select_torch_device, init_logging, is_torch_device_available
from lerobot.value.algorithms import (
    EpisodeTargetInfo,
    binarize_advantages,
    build_bin_centers,
    compute_n_step_advantages,
    compute_normalized_rewards_from_targets,
    compute_normalized_value_targets,
    compute_task_thresholds,
    expected_value_from_logits,
    project_values_to_bins,
    soft_cross_entropy,
)
from lerobot.value.io import (
    load_value_model_from_checkpoint,
    save_value_checkpoint,
    write_annotations_in_place,
)
from lerobot.value.modeling import make_value_model
from lerobot.value.telemetry import (
    build_diagnostics_payload,
    log_advantage_outputs,
    log_array_stats,
    log_checkpoint_saved,
    log_dataset_overview,
    log_inference_outputs,
    log_input_field_check,
    log_pipeline_summary,
    log_task_target_reward_stats,
    log_write_modes,
    save_diagnostics,
    stats_per_task,
    summarize_task_episodes,
)


def _resolve_device(requested_device: str | None) -> torch.device:
    if requested_device:
        try:
            if is_torch_device_available(requested_device):
                return torch.device(requested_device)
        except ValueError:
            logging.warning("Unsupported device '%s'. Falling back to auto selection.", requested_device)
        logging.warning("Device '%s' unavailable. Falling back to auto selection.", requested_device)
    return auto_select_torch_device()


def _build_episode_info(
    dataset: LeRobotDataset,
    success_field: str,
    default_success: str,
) -> tuple[dict[int, EpisodeTargetInfo], dict[int, int]]:
    episodes_ds = dataset.meta.episodes.with_format(None)
    episodes = episodes_ds[:]
    n_episodes = len(episodes_ds)
    has_success = success_field in episodes_ds.column_names

    episode_info: dict[int, EpisodeTargetInfo] = {}
    task_max_length: dict[int, int] = {}
    for i in range(n_episodes):
        ep_idx = int(episodes["episode_index"][i])
        ep_length = int(episodes["length"][i])
        tasks = episodes["tasks"][i]
        task_name = tasks[0] if isinstance(tasks, list) else tasks
        if task_name not in dataset.meta.tasks.index:
            raise KeyError(f"Episode {ep_idx} references unknown task '{task_name}'.")
        task_index = int(dataset.meta.tasks.loc[task_name].task_index)

        explicit_success = episodes[success_field][i] if has_success else None
        resolved_success = resolve_episode_success_label(
            explicit_success, default_label=default_success, require_label=True
        )
        ep_success = resolved_success == EPISODE_SUCCESS

        episode_info[ep_idx] = EpisodeTargetInfo(
            episode_index=ep_idx,
            task_index=task_index,
            length=ep_length,
            success=ep_success,
        )
        task_max_length[task_index] = max(task_max_length.get(task_index, 0), ep_length)
    return episode_info, task_max_length


def _build_wandb_run(cfg: ValueTrainPipelineConfig):
    if not cfg.wandb.enable or not cfg.wandb.project:
        return None
    try:
        import wandb
    except ImportError:
        logging.warning("WandB is not installed. Continue without WandB logging.")
        return None

    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=cfg.job_name,
        notes=cfg.wandb.notes,
        mode=cfg.wandb.mode if cfg.wandb.mode in ["online", "offline", "disabled"] else "online",
        dir=cfg.output_dir / "value",
        config=cfg.to_dict(),
        save_code=False,
    )
    logging.info("Track this run --> %s", run.get_url())
    return run


def run_value_training_pipeline(cfg: ValueTrainPipelineConfig) -> dict[str, Any]:
    cfg.validate()
    value_output_dir = cfg.output_dir / "value"
    value_output_dir.mkdir(parents=True, exist_ok=True)
    init_logging(log_file=value_output_dir / "value_train.log")
    logging.info(pformat(cfg.to_dict()))
    log_pipeline_summary(cfg)

    if cfg.seed is not None:
        set_seed(cfg.seed)

    device = _resolve_device(cfg.train.device)
    use_amp = bool(cfg.train.use_amp and device.type == "cuda")
    amp_scaler = torch.cuda.amp.GradScaler() if use_amp else None

    dataset = LeRobotDataset(
        repo_id=cfg.dataset.repo_id,
        root=cfg.dataset.root,
        episodes=cfg.dataset.episodes,
        revision=cfg.dataset.revision,
        download_videos=cfg.dataset.download_videos,
    )
    raw_frames = dataset.hf_dataset.with_format(None)
    frame_count = len(raw_frames)
    if frame_count == 0:
        raise ValueError("Dataset has no frames.")

    state_feature_exists = cfg.value.state_feature in dataset.hf_dataset.column_names
    task_feature_exists = cfg.value.task_index_feature in dataset.hf_dataset.column_names
    success_field_exists = cfg.dataset.success_field in dataset.meta.episodes.column_names
    intervention_field_exists = cfg.acp.intervention_field in dataset.hf_dataset.column_names
    log_input_field_check(
        cfg,
        state_feature_exists=state_feature_exists,
        task_feature_exists=task_feature_exists,
        success_field_exists=success_field_exists,
        intervention_field_exists=intervention_field_exists,
    )
    if not state_feature_exists:
        raise KeyError(f"Missing state feature '{cfg.value.state_feature}' in dataset columns.")
    if not task_feature_exists:
        raise KeyError(f"Missing task feature '{cfg.value.task_index_feature}' in dataset columns.")

    states = np.asarray(raw_frames[cfg.value.state_feature], dtype=np.float32)
    if states.ndim != 2:
        raise ValueError(
            f"Expected '{cfg.value.state_feature}' to be a rank-2 array [N, D], got shape={states.shape}."
        )
    task_indices = np.asarray(raw_frames[cfg.value.task_index_feature], dtype=np.int64)
    episode_indices = np.asarray(raw_frames["episode_index"], dtype=np.int64)
    frame_indices = np.asarray(raw_frames["frame_index"], dtype=np.int64)
    absolute_indices = np.asarray(raw_frames["index"], dtype=np.int64)

    if intervention_field_exists:
        interventions = np.asarray(raw_frames[cfg.acp.intervention_field], dtype=np.float32)
    else:
        interventions = np.zeros(frame_count, dtype=np.float32)

    episode_info, task_max_length = _build_episode_info(
        dataset=dataset,
        success_field=cfg.dataset.success_field,
        default_success=cfg.dataset.default_success,
    )
    value_targets = compute_normalized_value_targets(
        episode_indices=episode_indices,
        frame_indices=frame_indices,
        episode_info=episode_info,
        task_max_lengths=task_max_length,
        c_fail_coef=cfg.targets.c_fail_coef,
        clip_min=cfg.value.bin_min,
        clip_max=cfg.value.bin_max,
    )
    rewards = compute_normalized_rewards_from_targets(
        targets=value_targets,
        episode_indices=episode_indices,
        frame_indices=frame_indices,
    )

    num_tasks = int(len(dataset.meta.tasks))
    state_dim = int(states.shape[1])
    log_dataset_overview(frame_count, len(episode_info), num_tasks, state_dim)
    per_task_episode = summarize_task_episodes(episode_info, task_max_length)
    target_stats = log_array_stats("Value target (G_norm)", value_targets)
    reward_stats = log_array_stats("N-step reward basis", rewards)
    per_task_target_stats = stats_per_task(value_targets, task_indices)
    per_task_reward_stats = stats_per_task(rewards, task_indices)
    log_task_target_reward_stats(per_task_target_stats, per_task_reward_stats)

    model = make_value_model(cfg.value, state_dim=state_dim, num_tasks=num_tasks).to(device)

    train_dataset = TensorDataset(
        torch.from_numpy(states),
        torch.from_numpy(task_indices),
        torch.from_numpy(value_targets),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    train_iter = cycle(train_loader)

    if cfg.optimizer is None:
        raise ValueError("Optimizer config is missing.")
    optimizer = cfg.optimizer.build(model.parameters())
    scheduler = cfg.scheduler.build(optimizer, cfg.train.max_steps) if cfg.scheduler is not None else None
    grad_clip_norm = float(cfg.optimizer.grad_clip_norm)
    bin_centers = build_bin_centers(cfg.value.num_bins, cfg.value.bin_min, cfg.value.bin_max, device=device)

    wandb_run = _build_wandb_run(cfg)
    checkpoint_root = value_output_dir / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    value_cfg_payload = {
        "backbone": cfg.value.backbone,
        "model_config": draccus.encode(cfg.value),
        "num_bins": cfg.value.num_bins,
        "bin_min": cfg.value.bin_min,
        "bin_max": cfg.value.bin_max,
        "state_feature": cfg.value.state_feature,
        "task_index_feature": cfg.value.task_index_feature,
        "state_dim": state_dim,
        "num_tasks": num_tasks,
    }

    logging.info(
        "Start value training with %s backbone | num_frames=%s num_episodes=%s",
        cfg.value.backbone,
        frame_count,
        len(episode_info),
    )
    start_time = time.time()
    last_loss = 0.0
    last_mae = 0.0

    for step in range(1, cfg.train.max_steps + 1):
        batch_state, batch_task_idx, batch_target = next(train_iter)
        batch_state = batch_state.to(device=device, dtype=torch.float32, non_blocking=True)
        batch_task_idx = batch_task_idx.to(device=device, dtype=torch.long, non_blocking=True)
        batch_target = batch_target.to(device=device, dtype=torch.float32, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(batch_state, batch_task_idx)
            soft_target = project_values_to_bins(batch_target, bin_centers)
            loss = soft_cross_entropy(logits, soft_target)
            pred_value = expected_value_from_logits(logits, bin_centers)
            value_mae = (pred_value - batch_target).abs().mean()

        if use_amp:
            if amp_scaler is None:
                raise RuntimeError("AMP scaler is unexpectedly None while AMP is enabled.")
            amp_scaler.scale(loss).backward()
            amp_scaler.unscale_(optimizer)
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            amp_scaler.step(optimizer)
            amp_scaler.update()
        else:
            loss.backward()
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        last_loss = float(loss.item())
        last_mae = float(value_mae.item())

        if step % cfg.train.log_freq == 0 or step == 1:
            elapsed = time.time() - start_time
            steps_per_sec = step / max(elapsed, 1e-6)
            log_payload = {
                "step": step,
                "loss": last_loss,
                "value_mae": last_mae,
                "steps_per_sec": steps_per_sec,
            }
            logging.info(
                "step=%d loss=%.6f value_mae=%.6f steps/s=%.2f",
                step,
                last_loss,
                last_mae,
                steps_per_sec,
            )
            if wandb_run is not None:
                wandb_run.log(log_payload, step=step)

        should_save = (step % cfg.train.save_every == 0) or (step == cfg.train.max_steps)
        if should_save:
            training_meta = {
                "step": step,
                "max_steps": cfg.train.max_steps,
                "dataset_repo_id": cfg.dataset.repo_id,
                "loss": last_loss,
                "value_mae": last_mae,
            }
            save_value_checkpoint(
                checkpoint_root=checkpoint_root,
                step=step,
                total_steps=cfg.train.max_steps,
                model=model,
                value_config_payload=value_cfg_payload,
                training_meta=training_meta,
            )
            log_checkpoint_saved(step, checkpoint_root)

    if wandb_run is not None:
        wandb_run.finish()

    inference_model, _, _ = load_value_model_from_checkpoint(
        checkpoint_root=checkpoint_root, checkpoint_ref="last", device=device
    )
    inference_loader = DataLoader(
        TensorDataset(torch.from_numpy(states), torch.from_numpy(task_indices)),
        batch_size=max(cfg.train.batch_size, 64),
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for batch_state, batch_task_idx in inference_loader:
            batch_state = batch_state.to(device=device, dtype=torch.float32, non_blocking=True)
            batch_task_idx = batch_task_idx.to(device=device, dtype=torch.long, non_blocking=True)
            logits = inference_model(batch_state, batch_task_idx)
            batch_value = expected_value_from_logits(logits, bin_centers)
            predictions.append(batch_value.detach().cpu().numpy().astype(np.float32))
    predicted_values = np.concatenate(predictions, axis=0)
    inference_stats, inference_sample_first_5 = log_inference_outputs(predicted_values)

    advantages = compute_n_step_advantages(
        rewards=rewards,
        values=predicted_values,
        episode_indices=episode_indices,
        frame_indices=frame_indices,
        n_step=cfg.acp.n_step,
    )
    thresholds = compute_task_thresholds(
        task_indices=task_indices,
        advantages=advantages,
        positive_ratio=cfg.acp.positive_ratio,
    )
    indicators = binarize_advantages(
        task_indices=task_indices,
        advantages=advantages,
        thresholds=thresholds,
        interventions=interventions,
        force_intervention_positive=cfg.acp.force_intervention_positive,
    )
    (
        advantage_stats,
        per_task_adv_stats,
        indicator_positive_ratio_per_task,
    ) = log_advantage_outputs(advantages, task_indices, thresholds, indicators)

    columns = {
        cfg.acp.value_field: predicted_values.astype(np.float32),
        cfg.acp.advantage_field: advantages.astype(np.float32),
        cfg.acp.indicator_field: indicators.astype(np.int64),
    }
    feature_infos = {
        cfg.acp.value_field: {"dtype": "float32", "shape": (1,), "names": None},
        cfg.acp.advantage_field: {"dtype": "float32", "shape": (1,), "names": None},
        cfg.acp.indicator_field: {"dtype": "int64", "shape": (1,), "names": None},
    }
    write_modes = write_annotations_in_place(
        dataset_root=Path(dataset.root),
        frame_indices=absolute_indices,
        columns=columns,
        feature_infos=feature_infos,
    )
    log_write_modes(cfg, write_modes)

    positive_ratio_observed = float(np.mean(indicators.astype(np.float32)))
    logging.info(
        "Finished Value pipeline. indicator_positive_ratio=%.4f n_step=%d",
        positive_ratio_observed,
        cfg.acp.n_step,
    )
    diagnostics = build_diagnostics_payload(
        cfg=cfg,
        state_feature_exists=state_feature_exists,
        task_feature_exists=task_feature_exists,
        success_field_exists=success_field_exists,
        intervention_field_exists=intervention_field_exists,
        frame_count=frame_count,
        episode_count=len(episode_info),
        num_tasks=num_tasks,
        state_dim=state_dim,
        per_task_episode=per_task_episode,
        target_stats=target_stats,
        reward_stats=reward_stats,
        per_task_target_stats=per_task_target_stats,
        per_task_reward_stats=per_task_reward_stats,
        last_loss=last_loss,
        last_value_mae=last_mae,
        inference_stats=inference_stats,
        inference_sample_first_5=inference_sample_first_5,
        advantage_stats=advantage_stats,
        per_task_adv_stats=per_task_adv_stats,
        thresholds=thresholds,
        indicator_positive_ratio_global=positive_ratio_observed,
        indicator_positive_ratio_per_task=indicator_positive_ratio_per_task,
        write_modes=write_modes,
    )
    diagnostics_path = save_diagnostics(value_output_dir / "diagnostics", diagnostics)
    logging.info("Diagnostics saved | %s", diagnostics_path)

    return {
        "num_frames": int(frame_count),
        "num_episodes": int(len(episode_info)),
        "last_loss": float(last_loss),
        "last_value_mae": float(last_mae),
        "indicator_positive_ratio": positive_ratio_observed,
        "checkpoint_root": str(checkpoint_root),
        "diagnostics_path": str(diagnostics_path),
    }
