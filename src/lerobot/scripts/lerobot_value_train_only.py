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

import logging
import time
from pathlib import Path
from pprint import pformat
from typing import Any

import draccus
import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from torch.utils.data import DataLoader

from lerobot.configs import parser
from lerobot.configs.value_train import ValueTrainPipelineConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import cycle
from lerobot.utils.constants import OBS_IMAGES
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.random_utils import set_seed
from lerobot.utils.recording_annotations import EPISODE_SUCCESS, resolve_episode_success_label
from lerobot.utils.utils import init_logging
from lerobot.value.algorithms import (
    EpisodeTargetInfo,
    build_bin_centers,
    compute_normalized_rewards_from_targets,
    compute_normalized_value_targets,
    expected_value_from_logits,
    project_values_to_bins,
    soft_cross_entropy,
)
from lerobot.value.configuration import SiglipGemmaValueConfig
from lerobot.value.io import save_value_checkpoint
from lerobot.value.modeling import make_value_model
from lerobot.value.preprocess import SiglipGemmaValuePreprocessor
from lerobot.value.telemetry import (
    log_array_stats,
    log_checkpoint_saved,
    log_dataset_overview,
    log_input_field_check,
    log_pipeline_summary,
    log_task_target_reward_stats,
    stats_per_task,
    summarize_task_episodes,
)


def _create_accelerator(cfg: ValueTrainPipelineConfig, accelerator: Accelerator | None) -> Accelerator:
    if accelerator is not None:
        return accelerator
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    force_cpu = cfg.train.device == "cpu"
    return Accelerator(step_scheduler_with_optimizer=False, kwargs_handlers=[ddp_kwargs], cpu=force_cpu)


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
            explicit_success,
            default_label=default_success,
            require_label=True,
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


def _build_lookup_table(absolute_indices: np.ndarray, values: np.ndarray) -> np.ndarray:
    if absolute_indices.ndim != 1:
        raise ValueError(f"Expected absolute_indices rank-1, got shape {absolute_indices.shape}")
    if values.ndim != 1:
        raise ValueError(f"Expected values rank-1, got shape {values.shape}")
    if absolute_indices.shape[0] != values.shape[0]:
        raise ValueError(
            f"Lookup source size mismatch: absolute_indices={absolute_indices.shape[0]} "
            f"values={values.shape[0]}"
        )
    max_index = int(np.max(absolute_indices))
    table = np.zeros(max_index + 1, dtype=np.float32)
    table[absolute_indices] = values.astype(np.float32, copy=False)
    return table


def _build_wandb_run(cfg: ValueTrainPipelineConfig, accelerator: Accelerator):
    if not accelerator.is_main_process:
        return None
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


def _load_dataset_distributed(cfg: ValueTrainPipelineConfig, accelerator: Accelerator) -> LeRobotDataset:
    dataset_kwargs = {
        "repo_id": cfg.dataset.repo_id,
        "root": cfg.dataset.root,
        "episodes": cfg.dataset.episodes,
        "revision": cfg.dataset.revision,
        "download_videos": cfg.dataset.download_videos,
    }

    if accelerator.is_main_process:
        dataset = LeRobotDataset(**dataset_kwargs)
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        dataset = LeRobotDataset(**dataset_kwargs)
    return dataset


def _prepare_shared_runtime(
    cfg: ValueTrainPipelineConfig,
    dataset: LeRobotDataset,
    device: torch.device,
) -> dict[str, Any]:
    raw_frames = dataset.hf_dataset.with_format(None)
    frame_count = len(raw_frames)
    if frame_count == 0:
        raise ValueError("Dataset has no frames.")

    state_feature_exists = cfg.value.state_feature in dataset.hf_dataset.column_names
    task_feature_exists = cfg.value.task_index_feature in dataset.hf_dataset.column_names
    success_field_exists = cfg.dataset.success_field in dataset.meta.episodes.column_names
    intervention_field_exists = cfg.acp.intervention_field in dataset.hf_dataset.column_names
    task_field_exists = cfg.value.task_field == "task" or cfg.value.task_field in dataset.hf_dataset.column_names
    log_input_field_check(
        cfg,
        state_feature_exists=state_feature_exists,
        task_feature_exists=task_feature_exists,
        success_field_exists=success_field_exists,
        intervention_field_exists=intervention_field_exists,
    )
    if not task_feature_exists:
        raise KeyError(f"Missing task feature '{cfg.value.task_index_feature}' in dataset columns.")
    if not task_field_exists:
        raise KeyError(
            f"Missing task field '{cfg.value.task_field}'. "
            "Use `value.task_field=task` or provide an existing dataset column."
        )

    camera_features_available = [key for key in dataset.meta.camera_keys if key.startswith(OBS_IMAGES)]
    value_preprocessor = SiglipGemmaValuePreprocessor(
        cfg=cfg.value,
        dataset_camera_features=camera_features_available,
    )
    if not cfg.value.camera_features:
        cfg.value.camera_features = list(value_preprocessor.camera_features)
    logging.info(
        "Value preprocessor setup | task_field=%s cameras=%s",
        cfg.value.task_field,
        value_preprocessor.camera_features,
    )

    task_indices = np.asarray(raw_frames[cfg.value.task_index_feature], dtype=np.int64)
    episode_indices = np.asarray(raw_frames["episode_index"], dtype=np.int64)
    frame_indices = np.asarray(raw_frames["frame_index"], dtype=np.int64)
    absolute_indices = np.asarray(raw_frames["index"], dtype=np.int64)

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
    state_dim = 0
    log_dataset_overview(frame_count, len(episode_info), num_tasks, state_dim)
    per_task_episode = summarize_task_episodes(episode_info, task_max_length)
    log_array_stats("Value target (G_norm)", value_targets)
    log_array_stats("N-step reward basis", rewards)
    per_task_target_stats = stats_per_task(value_targets, task_indices)
    per_task_reward_stats = stats_per_task(rewards, task_indices)
    log_task_target_reward_stats(per_task_target_stats, per_task_reward_stats)

    bin_centers = build_bin_centers(cfg.value.num_bins, cfg.value.bin_min, cfg.value.bin_max, device=device)

    return {
        "frame_count": frame_count,
        "num_tasks": num_tasks,
        "state_dim": state_dim,
        "value_preprocessor": value_preprocessor,
        "absolute_indices": absolute_indices,
        "value_targets": value_targets,
        "episode_info": episode_info,
        "bin_centers": bin_centers,
    }


def _init_value_runtime(
    cfg: ValueTrainPipelineConfig,
    accelerator: Accelerator,
) -> tuple[Path, torch.device]:
    value_output_dir = cfg.output_dir / "value"
    if accelerator.is_main_process:
        value_output_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    log_file = value_output_dir / "value_train.log" if accelerator.is_main_process else None
    init_logging(log_file=log_file, accelerator=accelerator)

    if accelerator.is_main_process:
        logging.info(pformat(cfg.to_dict()))
        log_pipeline_summary(cfg)

    if cfg.seed is not None:
        set_seed(cfg.seed, accelerator=accelerator)

    device = accelerator.device
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    return value_output_dir, device


def run_value_training_only_pipeline(
    cfg: ValueTrainPipelineConfig,
    accelerator: Accelerator | None = None,
) -> dict[str, Any]:
    cfg.validate()
    if not isinstance(cfg.value, SiglipGemmaValueConfig):
        raise ValueError(
            f"Unsupported value config type '{cfg.value.type}'. "
            "This pipeline currently expects SiglipGemmaValueConfig."
        )

    accelerator = _create_accelerator(cfg, accelerator)
    value_output_dir, device = _init_value_runtime(cfg, accelerator)

    dataset = _load_dataset_distributed(cfg, accelerator)
    shared = _prepare_shared_runtime(cfg, dataset, device)

    model = make_value_model(cfg.value, state_dim=shared["state_dim"], num_tasks=shared["num_tasks"]).to(device)
    value_target_lookup = _build_lookup_table(
        absolute_indices=shared["absolute_indices"],
        values=shared["value_targets"],
    )

    train_loader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    if cfg.optimizer is None:
        raise ValueError("Optimizer config is missing.")
    optimizer = cfg.optimizer.build(model.parameters())
    scheduler = cfg.scheduler.build(optimizer, cfg.train.max_steps) if cfg.scheduler is not None else None

    if scheduler is None:
        model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    else:
        model, optimizer, train_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, scheduler)

    train_iter = cycle(train_loader)
    grad_clip_norm = float(cfg.optimizer.grad_clip_norm)
    wandb_run = _build_wandb_run(cfg, accelerator)

    checkpoint_root = value_output_dir / "checkpoints"
    if accelerator.is_main_process:
        checkpoint_root.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    value_cfg_payload = {
        "backbone": cfg.value.backbone,
        "model_config": draccus.encode(cfg.value),
        "num_bins": cfg.value.num_bins,
        "bin_min": cfg.value.bin_min,
        "bin_max": cfg.value.bin_max,
        "state_feature": cfg.value.state_feature,
        "task_index_feature": cfg.value.task_index_feature,
        "state_dim": shared["state_dim"],
        "num_tasks": shared["num_tasks"],
    }

    if accelerator.is_main_process:
        logging.info(
            "Start distributed value training | world_size=%d backbone=%s num_frames=%s num_episodes=%s",
            accelerator.num_processes,
            cfg.value.backbone,
            shared["frame_count"],
            len(shared["episode_info"]),
        )

    start_time = time.time()
    last_loss = 0.0
    last_mae = 0.0

    for step in range(1, cfg.train.max_steps + 1):
        raw_batch = next(train_iter)
        model_inputs = shared["value_preprocessor"](raw_batch, device=device)
        batch_indices = raw_batch["index"]
        if not isinstance(batch_indices, torch.Tensor):
            batch_indices = torch.as_tensor(batch_indices)
        batch_indices_np = batch_indices.detach().cpu().numpy().astype(np.int64, copy=False).reshape(-1)
        batch_target = torch.from_numpy(value_target_lookup[batch_indices_np]).to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )

        optimizer.zero_grad(set_to_none=True)
        with accelerator.autocast():
            logits = model(**model_inputs)
            soft_target = project_values_to_bins(batch_target, shared["bin_centers"])
            loss = soft_cross_entropy(logits, soft_target)
            pred_value = expected_value_from_logits(logits, shared["bin_centers"])
            value_mae = (pred_value - batch_target).abs().mean()

        accelerator.backward(loss)
        if grad_clip_norm > 0:
            accelerator.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        last_loss = float(loss.detach().item())
        last_mae = float(value_mae.detach().item())

        if accelerator.is_main_process and (step % cfg.train.log_freq == 0 or step == 1):
            elapsed = time.time() - start_time
            steps_per_sec = step / max(elapsed, 1e-6)
            logging.info(
                "step=%d loss=%.6f value_mae=%.6f steps/s=%.2f",
                step,
                last_loss,
                last_mae,
                steps_per_sec,
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "step": step,
                        "loss": last_loss,
                        "value_mae": last_mae,
                        "steps_per_sec": steps_per_sec,
                    },
                    step=step,
                )

        should_save = (step % cfg.train.save_every == 0) or (step == cfg.train.max_steps)
        if should_save and accelerator.is_main_process:
            training_meta = {
                "step": step,
                "max_steps": cfg.train.max_steps,
                "dataset_repo_id": cfg.dataset.repo_id,
                "loss": last_loss,
                "value_mae": last_mae,
            }
            model_to_save = accelerator.unwrap_model(model, keep_fp32_wrapper=True)
            save_value_checkpoint(
                checkpoint_root=checkpoint_root,
                step=step,
                total_steps=cfg.train.max_steps,
                model=model_to_save,
                value_config_payload=value_cfg_payload,
                training_meta=training_meta,
            )
            log_checkpoint_saved(step, checkpoint_root)
        if should_save:
            accelerator.wait_for_everyone()

    if wandb_run is not None:
        wandb_run.finish()

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        return {
            "num_frames": int(shared["frame_count"]),
            "num_episodes": int(len(shared["episode_info"])),
            "last_loss": float(last_loss),
            "last_value_mae": float(last_mae),
            "checkpoint_root": str(checkpoint_root),
            "world_size": int(accelerator.num_processes),
        }
    return {"world_size": int(accelerator.num_processes), "main_process": False}


@parser.wrap()
def value_train_only(cfg: ValueTrainPipelineConfig):
    return run_value_training_only_pipeline(cfg)


def main():
    register_third_party_plugins()
    value_train_only()


if __name__ == "__main__":
    main()
