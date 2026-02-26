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
import math
from pathlib import Path
from pprint import pformat
from typing import Any

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from huggingface_hub import snapshot_download
from torch.utils.data import DataLoader, Sampler
from tqdm.auto import tqdm

from lerobot.configs import parser
from lerobot.configs.value import ValueInferencePipelineConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import OBS_IMAGES
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.random_utils import set_seed
from lerobot.utils.recording_annotations import EPISODE_SUCCESS, resolve_episode_success_label
from lerobot.utils.utils import init_logging, inside_slurm
from lerobot.value.algorithms import (
    EpisodeTargetInfo,
    binarize_advantages,
    build_bin_centers,
    compute_n_step_advantages,
    compute_normalized_rewards_from_targets,
    compute_normalized_value_targets,
    compute_task_thresholds,
    expected_value_from_logits,
)
from lerobot.value.configuration import SiglipGemmaValueConfig
from lerobot.value.io import load_value_model_from_checkpoint, write_annotations_in_place
from lerobot.value.preprocess import SiglipGemmaValuePreprocessor
from lerobot.value.telemetry import (
    build_diagnostics_payload,
    log_advantage_outputs,
    log_array_stats,
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


class ContiguousDistributedEvalSampler(Sampler[int]):
    """Distributed eval sampler with contiguous per-rank shards and deterministic tail padding."""

    def __init__(self, dataset_size: int, num_replicas: int, rank: int):
        if dataset_size <= 0:
            raise ValueError(f"'dataset_size' must be > 0, got {dataset_size}.")
        if num_replicas <= 0:
            raise ValueError(f"'num_replicas' must be > 0, got {num_replicas}.")
        if rank < 0 or rank >= num_replicas:
            raise ValueError(f"'rank' must be in [0, {num_replicas - 1}], got {rank}.")

        self.dataset_size = int(dataset_size)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.num_samples = int(math.ceil(self.dataset_size / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        shard_start = self.rank * self.num_samples
        shard_end = min(shard_start + self.num_samples, self.dataset_size)
        if shard_start >= self.dataset_size:
            indices: list[int] = []
        else:
            indices = list(range(shard_start, shard_end))

        if len(indices) < self.num_samples:
            indices.extend([self.dataset_size - 1] * (self.num_samples - len(indices)))
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples


def _set_infer_logger_levels() -> None:
    for logger_name in ["fsspec", "fsspec.local", "huggingface_hub", "datasets", "torchcodec"]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def _create_accelerator(cfg: ValueInferencePipelineConfig, accelerator: Accelerator | None) -> Accelerator:
    if accelerator is not None:
        return accelerator
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    force_cpu = cfg.runtime.device == "cpu"
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


def _load_dataset_distributed(cfg: ValueInferencePipelineConfig, accelerator: Accelerator) -> LeRobotDataset:
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


def _resolve_checkpoint_source(
    cfg: ValueInferencePipelineConfig, value_output_dir: Path
) -> tuple[Path, str]:
    if cfg.inference.repo_id:
        checkpoint_root = Path(
            snapshot_download(
                repo_id=cfg.inference.repo_id,
                repo_type="model",
                revision=cfg.inference.revision,
            )
        )
        logging.info("Loaded checkpoint source from hub repo | repo_id=%s local=%s", cfg.inference.repo_id, checkpoint_root)
    elif cfg.inference.checkpoint_root is not None:
        checkpoint_root = Path(cfg.inference.checkpoint_root)
        logging.info("Loaded checkpoint source from local path | root=%s", checkpoint_root)
    else:
        checkpoint_root = value_output_dir / "checkpoints"
        logging.info("Loaded checkpoint source from output dir | root=%s", checkpoint_root)

    if not checkpoint_root.exists():
        raise FileNotFoundError(f"Checkpoint root not found: {checkpoint_root}")

    checkpoint_ref = cfg.inference.checkpoint_ref
    if checkpoint_ref == "last" and not (checkpoint_root / "last").exists():
        # Support flattened checkpoints uploaded via:
        # hf upload <repo> <step_dir> . --repo-type model
        if (checkpoint_root / "weights.pt").is_file() and (checkpoint_root / "value_config.json").is_file():
            checkpoint_ref = "."
            logging.info("Checkpoint root is flattened; fallback checkpoint_ref='.'")
        else:
            raise FileNotFoundError(
                f"checkpoint_ref='last' but {checkpoint_root / 'last'} is missing. "
                "Set --inference.checkpoint_ref explicitly."
            )
    return checkpoint_root, checkpoint_ref


def _prepare_shared_runtime(
    cfg: ValueInferencePipelineConfig,
    dataset: LeRobotDataset,
    model_cfg: SiglipGemmaValueConfig,
    value_config_payload: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    raw_frames = dataset.hf_dataset.with_format(None)
    frame_count = len(raw_frames)
    if frame_count == 0:
        raise ValueError("Dataset has no frames.")

    state_feature = str(value_config_payload.get("state_feature", model_cfg.state_feature))
    task_index_feature = str(value_config_payload.get("task_index_feature", model_cfg.task_index_feature))
    task_field = str(model_cfg.task_field)
    num_bins = int(value_config_payload.get("num_bins", model_cfg.num_bins))
    bin_min = float(value_config_payload.get("bin_min", model_cfg.bin_min))
    bin_max = float(value_config_payload.get("bin_max", model_cfg.bin_max))

    state_feature_exists = state_feature in dataset.hf_dataset.column_names
    task_feature_exists = task_index_feature in dataset.hf_dataset.column_names
    success_field_exists = cfg.dataset.success_field in dataset.meta.episodes.column_names
    intervention_field_exists = cfg.acp.intervention_field in dataset.hf_dataset.column_names
    task_field_exists = task_field == "task" or task_field in dataset.hf_dataset.column_names
    log_input_field_check(
        cfg,
        state_feature=state_feature,
        task_feature=task_index_feature,
        state_feature_exists=state_feature_exists,
        task_feature_exists=task_feature_exists,
        success_field_exists=success_field_exists,
        intervention_field_exists=intervention_field_exists,
    )
    if not task_feature_exists:
        raise KeyError(f"Missing task feature '{task_index_feature}' in dataset columns.")
    if not task_field_exists:
        raise KeyError(
            f"Missing task field '{task_field}'. "
            "Set a valid task field in the trained value config."
        )

    camera_features_available = [key for key in dataset.meta.camera_keys if key.startswith(OBS_IMAGES)]
    value_preprocessor = SiglipGemmaValuePreprocessor(
        cfg=model_cfg,
        dataset_camera_features=camera_features_available,
    )
    if not model_cfg.camera_features:
        model_cfg.camera_features = list(value_preprocessor.camera_features)
    logging.info(
        "Value preprocessor setup | task_field=%s cameras=%s",
        task_field,
        value_preprocessor.camera_features,
    )

    task_indices = np.asarray(raw_frames[task_index_feature], dtype=np.int64)
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
        clip_min=bin_min,
        clip_max=bin_max,
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
    target_stats = log_array_stats("Value target (G_norm)", value_targets)
    reward_stats = log_array_stats("N-step reward basis", rewards)
    per_task_target_stats = stats_per_task(value_targets, task_indices)
    per_task_reward_stats = stats_per_task(rewards, task_indices)
    log_task_target_reward_stats(per_task_target_stats, per_task_reward_stats)

    bin_centers = build_bin_centers(num_bins, bin_min, bin_max, device=device)

    return {
        "frame_count": frame_count,
        "num_tasks": num_tasks,
        "state_dim": state_dim,
        "state_feature": state_feature,
        "task_index_feature": task_index_feature,
        "value_preprocessor": value_preprocessor,
        "task_indices": task_indices,
        "episode_indices": episode_indices,
        "frame_indices": frame_indices,
        "absolute_indices": absolute_indices,
        "interventions": interventions,
        "rewards": rewards,
        "episode_info": episode_info,
        "per_task_episode": per_task_episode,
        "target_stats": target_stats,
        "reward_stats": reward_stats,
        "per_task_target_stats": per_task_target_stats,
        "per_task_reward_stats": per_task_reward_stats,
        "state_feature_exists": state_feature_exists,
        "task_feature_exists": task_feature_exists,
        "success_field_exists": success_field_exists,
        "intervention_field_exists": intervention_field_exists,
        "bin_centers": bin_centers,
    }


def _init_value_runtime(
    cfg: ValueInferencePipelineConfig,
    accelerator: Accelerator,
) -> tuple[Path, torch.device]:
    value_output_dir = cfg.output_dir / "value"
    if accelerator.is_main_process:
        value_output_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    log_file = value_output_dir / "value_infer.log" if accelerator.is_main_process else None
    init_logging(log_file=log_file, file_level="INFO", accelerator=accelerator)
    _set_infer_logger_levels()

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


def _push_dataset_to_hub_if_enabled(cfg: ValueInferencePipelineConfig, dataset: LeRobotDataset) -> bool:
    if not cfg.push_to_hub:
        return False
    logging.info(
        "Start pushing inferred dataset to hub | repo_id=%s root=%s push_videos=%s",
        dataset.repo_id,
        dataset.root,
        False,
    )
    dataset.push_to_hub(push_videos=False)
    logging.info("Finished pushing inferred dataset to hub | repo_id=%s", dataset.repo_id)
    return True


def run_value_inference_only_pipeline(
    cfg: ValueInferencePipelineConfig,
    accelerator: Accelerator | None = None,
) -> dict[str, Any]:
    cfg.validate()

    accelerator = _create_accelerator(cfg, accelerator)
    value_output_dir, device = _init_value_runtime(cfg, accelerator)

    dataset = _load_dataset_distributed(cfg, accelerator)
    checkpoint_root, checkpoint_ref = _resolve_checkpoint_source(cfg, value_output_dir)
    inference_model, model_cfg, value_config_payload = load_value_model_from_checkpoint(
        checkpoint_root=checkpoint_root,
        checkpoint_ref=checkpoint_ref,
        device=device,
    )
    if not isinstance(model_cfg, SiglipGemmaValueConfig):
        raise ValueError(
            f"Unsupported value config type '{type(model_cfg)}'. "
            "This pipeline currently expects SiglipGemmaValueConfig."
        )

    shared = _prepare_shared_runtime(cfg, dataset, model_cfg, value_config_payload, device)
    inference_model.eval()

    inference_sampler = ContiguousDistributedEvalSampler(
        dataset_size=len(dataset),
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
    )
    inference_loader = DataLoader(
        dataset,
        batch_size=max(cfg.runtime.batch_size, 64),
        shuffle=False,
        sampler=inference_sampler,
        num_workers=cfg.runtime.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    inference_model = accelerator.prepare(inference_model)

    if accelerator.is_main_process:
        max_abs_index = int(np.max(shared["absolute_indices"]))
        prediction_lookup = np.zeros(max_abs_index + 1, dtype=np.float32)
        prediction_seen = np.zeros(max_abs_index + 1, dtype=np.bool_)
    else:
        prediction_lookup = None
        prediction_seen = None

    if accelerator.is_main_process:
        logging.info(
            "Contiguous eval sampler | num_samples_per_rank=%d total_size=%d dataset_size=%d",
            inference_sampler.num_samples,
            inference_sampler.total_size,
            len(dataset),
        )
        logging.info(
            "Start distributed value inference | world_size=%d local_batches=%d local_batch_size=%s",
            accelerator.num_processes,
            len(inference_loader),
            inference_loader.batch_size,
        )

    inference_iter = tqdm(
        inference_loader,
        desc="Value inference",
        total=len(inference_loader),
        leave=False,
        disable=(not accelerator.is_main_process) or inside_slurm(),
    )

    with torch.no_grad():
        for raw_batch in inference_iter:
            model_inputs = shared["value_preprocessor"](raw_batch, device=device)
            batch_indices = raw_batch["index"]
            if not isinstance(batch_indices, torch.Tensor):
                batch_indices = torch.as_tensor(batch_indices)
            batch_indices = batch_indices.to(device=device, dtype=torch.long, non_blocking=True)

            with accelerator.autocast():
                logits = inference_model(**model_inputs)
                batch_value = expected_value_from_logits(logits, shared["bin_centers"])

            gathered_idx = accelerator.gather_for_metrics(batch_indices)
            gathered_val = accelerator.gather_for_metrics(batch_value)

            if accelerator.is_main_process and prediction_lookup is not None and prediction_seen is not None:
                idx_np = gathered_idx.detach().cpu().numpy().astype(np.int64, copy=False).reshape(-1)
                val_np = gathered_val.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
                prediction_lookup[idx_np] = val_np
                prediction_seen[idx_np] = True

    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        result: dict[str, Any] = {"world_size": int(accelerator.num_processes), "main_process": False}
    else:
        if prediction_lookup is None or prediction_seen is None:
            raise RuntimeError("Prediction buffers unexpectedly missing on main process.")

        missing_mask = ~prediction_seen[shared["absolute_indices"]]
        if bool(np.any(missing_mask)):
            missing_count = int(np.sum(missing_mask))
            raise RuntimeError(f"Inference is missing predictions for {missing_count} frames.")

        predicted_values = prediction_lookup[shared["absolute_indices"]]
        inference_stats, inference_sample_first_5 = log_inference_outputs(predicted_values)

        advantages = compute_n_step_advantages(
            rewards=shared["rewards"],
            values=predicted_values,
            episode_indices=shared["episode_indices"],
            frame_indices=shared["frame_indices"],
            n_step=cfg.acp.n_step,
        )
        thresholds = compute_task_thresholds(
            task_indices=shared["task_indices"],
            advantages=advantages,
            positive_ratio=cfg.acp.positive_ratio,
        )
        indicators = binarize_advantages(
            task_indices=shared["task_indices"],
            advantages=advantages,
            thresholds=thresholds,
            interventions=shared["interventions"],
            force_intervention_positive=cfg.acp.force_intervention_positive,
        )
        (
            advantage_stats,
            per_task_adv_stats,
            indicator_positive_ratio_per_task,
        ) = log_advantage_outputs(advantages, shared["task_indices"], thresholds, indicators)

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
            frame_indices=shared["absolute_indices"],
            columns=columns,
            feature_infos=feature_infos,
        )
        log_write_modes(cfg, write_modes)

        positive_ratio_observed = float(np.mean(indicators.astype(np.float32)))
        logging.info(
            "Finished distributed value inference. indicator_positive_ratio=%.4f n_step=%d",
            positive_ratio_observed,
            cfg.acp.n_step,
        )
        diagnostics = build_diagnostics_payload(
            cfg=cfg,
            state_feature=shared["state_feature"],
            task_feature=shared["task_index_feature"],
            state_feature_exists=shared["state_feature_exists"],
            task_feature_exists=shared["task_feature_exists"],
            success_field_exists=shared["success_field_exists"],
            intervention_field_exists=shared["intervention_field_exists"],
            frame_count=shared["frame_count"],
            episode_count=len(shared["episode_info"]),
            num_tasks=shared["num_tasks"],
            state_dim=shared["state_dim"],
            per_task_episode=shared["per_task_episode"],
            target_stats=shared["target_stats"],
            reward_stats=shared["reward_stats"],
            per_task_target_stats=shared["per_task_target_stats"],
            per_task_reward_stats=shared["per_task_reward_stats"],
            last_loss=float("nan"),
            last_value_mae=float("nan"),
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
        dataset_pushed_to_hub = _push_dataset_to_hub_if_enabled(cfg, dataset)

        result = {
            "num_frames": int(shared["frame_count"]),
            "num_episodes": int(len(shared["episode_info"])),
            "indicator_positive_ratio": positive_ratio_observed,
            "checkpoint_root": str(checkpoint_root),
            "checkpoint_ref": checkpoint_ref,
            "diagnostics_path": str(diagnostics_path),
            "dataset_pushed_to_hub": dataset_pushed_to_hub,
            "world_size": int(accelerator.num_processes),
            "main_process": True,
        }

    del inference_iter
    del inference_loader
    accelerator.wait_for_everyone()
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    return result


@parser.wrap()
def value_infer_only(cfg: ValueInferencePipelineConfig):
    return run_value_inference_only_pipeline(cfg)


def main():
    register_third_party_plugins()
    value_infer_only()


if __name__ == "__main__":
    main()
