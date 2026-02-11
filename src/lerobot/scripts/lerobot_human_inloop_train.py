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

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from huggingface_hub import HfApi

from lerobot.configs import parser
from lerobot.configs.default import WandBConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_policy_config, make_pre_post_processors
from lerobot.rl.recap_value import (
    MLPValueModel,
    PI05ValueModel,
    build_episode_value_info,
    build_normalized_returns,
    configure_pi05_trainable_params,
    expected_value_from_logits,
    project_values_to_bins,
    soft_cross_entropy,
)
from lerobot.utils.constants import OBS_STATE
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import init_logging


@dataclass
class HumanInloopDatasetConfig:
    repo_id: str = ""
    root: str | None = None
    episodes: str | None = None
    download_videos: bool = False
    success_field: str = "episode_success"
    default_success: str = "failure"


@dataclass
class HumanInloopValueConfig:
    backbone: str = "mlp"
    pretrained_path: str = "lerobot/pi05_base"
    num_bins: int = 201
    c_fail: float = 10.0
    hidden_dim: int = 512
    dropout: float = 0.1
    freeze_backbone: bool = True
    finetune_last_n_layers: int = 0
    save_backbone: bool = False
    push_to_hub: bool = False
    repo_id: str = ""
    private: bool | None = None
    hub_branch: str | None = None
    hub_create_pr: bool | None = None
    hub_commit_message: str | None = None


@dataclass
class HumanInloopPolicyConfig:
    push_to_hub: bool = False
    repo_id: str = ""
    private: bool | None = None


@dataclass
class HumanInloopTrainRuntimeConfig:
    device: str = "cuda"
    seed: int = 42
    batch_size: int = 8
    num_workers: int = 0
    lr: float = 1e-4
    weight_decay: float = 1e-5
    max_steps: int = 1000
    log_freq: int = 20
    save_every: int = 200


@dataclass
class HumanInloopTrainConfig:
    stage: str = "value"
    job_name: str | None = None
    output_dir: str = "outputs/human_inloop_train"
    dataset: HumanInloopDatasetConfig = field(default_factory=HumanInloopDatasetConfig)
    value: HumanInloopValueConfig = field(default_factory=HumanInloopValueConfig)
    policy: HumanInloopPolicyConfig = field(default_factory=HumanInloopPolicyConfig)
    train: HumanInloopTrainRuntimeConfig = field(default_factory=HumanInloopTrainRuntimeConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)


def _parse_episodes_arg(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    text = raw.strip()
    if not text:
        return None
    if ":" in text:
        parts = text.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid episodes range format: {raw!r}")
        start, end = int(parts[0]), int(parts[1])
        if end <= start:
            raise ValueError(f"episodes range must satisfy end > start, got {raw!r}")
        return list(range(start, end))
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def _resolve_device(raw_device: str) -> torch.device:
    if raw_device.startswith("cuda") and not torch.cuda.is_available():
        logging.warning("CUDA is unavailable, falling back to CPU.")
        return torch.device("cpu")
    return torch.device(raw_device)


def _move_tensor_batch_to_device(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def _infer_state_dim(dataset: LeRobotDataset) -> int:
    if OBS_STATE not in dataset.meta.features:
        raise ValueError(f"Dataset is missing required feature: {OBS_STATE}")
    shape = dataset.meta.features[OBS_STATE]["shape"]
    dim = 1
    for axis in shape:
        dim *= int(axis)
    return dim


def _build_value_model(
    cfg: HumanInloopTrainConfig,
    dataset: LeRobotDataset,
    device: torch.device,
):
    if cfg.value.backbone == "mlp":
        state_dim = _infer_state_dim(dataset)
        model = MLPValueModel(
            state_dim=state_dim,
            num_tasks=max(1, dataset.meta.total_tasks),
            hidden_dim=cfg.value.hidden_dim,
            num_bins=cfg.value.num_bins,
            dropout=cfg.value.dropout,
        ).to(device)
        return model, None

    if cfg.value.backbone != "pi05":
        raise ValueError(
            f"Unsupported value.backbone={cfg.value.backbone!r}. Expected one of: 'mlp', 'pi05'."
        )

    policy_cfg = make_policy_config(
        "pi05",
        device=device.type,
        pretrained_path=cfg.value.pretrained_path,
    )
    policy = make_policy(cfg=policy_cfg, ds_meta=dataset.meta)
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy_cfg,
        dataset_stats=dataset.meta.stats,
    )
    model = PI05ValueModel(pi05_policy=policy, num_bins=cfg.value.num_bins, dropout=cfg.value.dropout).to(device)
    configure_pi05_trainable_params(
        model=model,
        freeze_backbone=cfg.value.freeze_backbone,
        finetune_last_n_layers=cfg.value.finetune_last_n_layers,
    )
    return model, preprocessor


def _extract_checkpoint_state(model: torch.nn.Module, cfg: HumanInloopTrainConfig) -> dict:
    if cfg.value.backbone != "pi05":
        return {"model": model.state_dict()}

    pi05_model = model
    if cfg.value.save_backbone:
        return {"model": pi05_model.state_dict()}

    state = {
        "value_head": pi05_model.value_head.state_dict(),  # type: ignore[attr-defined]
    }
    trainable_backbone = {
        name: param.detach().cpu()
        for name, param in pi05_model.named_parameters()
        if param.requires_grad and not name.startswith("value_head.")
    }
    if trainable_backbone:
        state["trainable_backbone"] = trainable_backbone
    return state


def _save_checkpoint(
    cfg: HumanInloopTrainConfig,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    output_dir: Path,
) -> Path:
    checkpoint_dir = output_dir / "checkpoints" / f"step_{step:06d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "step": step,
        "optimizer": optimizer.state_dict(),
        "config": asdict(cfg),
    }
    checkpoint.update(_extract_checkpoint_state(model, cfg))
    torch.save(checkpoint, checkpoint_dir / "value_model.pt")

    with (output_dir / "checkpoints" / "last").open("w", encoding="utf-8") as f:
        f.write(f"step_{step:06d}\n")
    return checkpoint_dir


def _maybe_init_wandb(cfg: HumanInloopTrainConfig, output_dir: Path):
    if not cfg.wandb.enable or not cfg.wandb.project:
        logging.info("WandB is disabled. Logs will be saved locally.")
        return None

    os.environ["WANDB_SILENT"] = "True"
    import wandb

    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=cfg.job_name,
        notes=cfg.wandb.notes,
        dir=str(output_dir),
        config=asdict(cfg),
        save_code=False,
        mode=cfg.wandb.mode if cfg.wandb.mode in {"online", "offline", "disabled"} else "online",
    )
    cfg.wandb.run_id = run.id
    run_url = run.get_url()
    if run_url:
        logging.info("Track this run --> %s", run_url)
    else:
        logging.info("WandB run initialized (mode=%s, run_id=%s).", cfg.wandb.mode, run.id)
    return run


def _safe_artifact_name(name: str) -> str:
    return name.replace(":", "_").replace("/", "_")


def _maybe_log_wandb_checkpoint(cfg: HumanInloopTrainConfig, wandb_run, checkpoint_dir: Path) -> None:
    if wandb_run is None or cfg.wandb.disable_artifact:
        return

    model_path = checkpoint_dir / "value_model.pt"
    if not model_path.exists():
        logging.warning("Checkpoint file does not exist for wandb artifact: %s", model_path)
        return

    import wandb

    artifact_name = _safe_artifact_name(f"hil-value-{cfg.value.backbone}-{checkpoint_dir.name}")
    artifact = wandb.Artifact(artifact_name, type="model")
    artifact.add_file(str(model_path))
    wandb_run.log_artifact(artifact)


def _maybe_push_value_to_hub(cfg: HumanInloopTrainConfig, output_dir: Path) -> None:
    if not cfg.value.push_to_hub:
        return
    if not cfg.value.repo_id:
        raise ValueError("`value.repo_id` must be set when `value.push_to_hub=true`.")

    api = HfApi()
    repo_id = api.create_repo(
        repo_id=cfg.value.repo_id,
        private=cfg.value.private,
        exist_ok=True,
    ).repo_id
    commit_info = api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(output_dir),
        commit_message=cfg.value.hub_commit_message or "Upload value model checkpoints and config",
        revision=cfg.value.hub_branch,
        create_pr=cfg.value.hub_create_pr,
        ignore_patterns=["*.tmp", "*.log"],
    )
    repo_url = getattr(getattr(commit_info, "repo_url", None), "url", f"https://huggingface.co/{repo_id}")
    logging.info("Value model pushed to %s", repo_url)


def train_value(cfg: HumanInloopTrainConfig) -> None:
    if not cfg.dataset.repo_id:
        raise ValueError("dataset.repo_id must be provided for value training.")

    device = _resolve_device(cfg.train.device)
    set_seed(cfg.train.seed)

    episodes = _parse_episodes_arg(cfg.dataset.episodes)
    dataset = LeRobotDataset(
        repo_id=cfg.dataset.repo_id,
        root=cfg.dataset.root,
        episodes=episodes,
        download_videos=cfg.dataset.download_videos,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        drop_last=False,
        prefetch_factor=2 if cfg.train.num_workers > 0 else None,
    )
    if len(dataloader) == 0:
        raise ValueError("Dataloader is empty. Check dataset or episodes selection.")

    episode_info, task_lmax = build_episode_value_info(
        dataset.meta.episodes,
        success_field=cfg.dataset.success_field,
        default_success=cfg.dataset.default_success,
    )
    model, preprocessor = _build_value_model(cfg, dataset, device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters found in value model.")
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )

    output_dir = Path(cfg.output_dir).expanduser().resolve() / "value"
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)

    wandb_run = _maybe_init_wandb(cfg=cfg, output_dir=output_dir)

    logging.info("Start value training with %s backbone", cfg.value.backbone)
    logging.info("dataset.num_frames=%s, dataset.num_episodes=%s", dataset.num_frames, dataset.num_episodes)
    logging.info("output_dir=%s", output_dir)

    try:
        step = 0
        data_iter = iter(dataloader)
        while step < cfg.train.max_steps:
            try:
                raw_batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                raw_batch = next(data_iter)

            returns = build_normalized_returns(
                frame_indices=raw_batch["frame_index"].cpu(),
                episode_indices=raw_batch["episode_index"].cpu(),
                episode_info=episode_info,
                task_lmax=task_lmax,
                c_fail=cfg.value.c_fail,
            ).to(device)
            targets, bins = project_values_to_bins(returns, cfg.value.num_bins)

            if preprocessor is not None:
                batch = preprocessor(raw_batch)
            else:
                batch = _move_tensor_batch_to_device(raw_batch, device)

            logits = model(batch)
            loss = soft_cross_entropy(logits, targets)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            step += 1

            if step % cfg.train.log_freq == 0 or step == 1:
                with torch.no_grad():
                    pred_value = expected_value_from_logits(logits, bins)
                    value_mae = torch.mean(torch.abs(pred_value - returns))
                lr = float(optimizer.param_groups[0]["lr"])
                logging.info("step=%d loss=%.6f value_mae=%.6f", step, loss.item(), value_mae.item())
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "train/loss": float(loss.item()),
                            "train/value_mae": float(value_mae.item()),
                            "train/lr": lr,
                        },
                        step=step,
                    )

            if step % cfg.train.save_every == 0 or step == cfg.train.max_steps:
                checkpoint_dir = _save_checkpoint(
                    cfg=cfg, model=model, optimizer=optimizer, step=step, output_dir=output_dir
                )
                _maybe_log_wandb_checkpoint(cfg=cfg, wandb_run=wandb_run, checkpoint_dir=checkpoint_dir)

        _maybe_push_value_to_hub(cfg=cfg, output_dir=output_dir)
    finally:
        if wandb_run is not None:
            wandb_run.finish()


@parser.wrap()
def human_inloop_train(cfg: HumanInloopTrainConfig):
    if cfg.stage != "value":
        raise NotImplementedError(
            f"`stage={cfg.stage}` is not implemented yet. "
            "Please run with `--stage=value` to train value function first."
        )
    train_value(cfg)


def main():
    init_logging()
    register_third_party_plugins()
    human_inloop_train()


if __name__ == "__main__":
    main()
