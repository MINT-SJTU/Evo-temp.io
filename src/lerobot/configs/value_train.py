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

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import draccus

from lerobot.configs.default import DatasetConfig, WandBConfig
from lerobot.optim.optimizers import OptimizerConfig
from lerobot.optim.schedulers import LRSchedulerConfig
from lerobot.utils.recording_annotations import normalize_episode_success_label
from lerobot.value.configuration import SiglipGemmaValueConfig, ValueModelConfig


@dataclass
class ValueDatasetConfig(DatasetConfig):
    download_videos: bool = True
    success_field: str = "episode_success"
    default_success: str = "failure"


@dataclass
class ValueTargetsConfig:
    c_fail_coef: float = 1.0

    def validate(self) -> None:
        if self.c_fail_coef < 0:
            raise ValueError("'targets.c_fail_coef' must be non-negative.")


@dataclass
class ValueAdvantageConfig:
    n_step: int = 50
    positive_ratio: float = 0.30
    value_field: str = "complementary_info.value"
    advantage_field: str = "complementary_info.advantage"
    indicator_field: str = "complementary_info.acp_indicator"
    force_intervention_positive: bool = True
    intervention_field: str = "complementary_info.is_intervention"

    def validate(self) -> None:
        if self.n_step <= 0:
            raise ValueError("'acp.n_step' must be > 0.")
        if not 0.0 <= self.positive_ratio <= 1.0:
            raise ValueError("'acp.positive_ratio' must be within [0, 1].")
        if not self.value_field:
            raise ValueError("'acp.value_field' must be non-empty.")
        if not self.advantage_field:
            raise ValueError("'acp.advantage_field' must be non-empty.")
        if not self.indicator_field:
            raise ValueError("'acp.indicator_field' must be non-empty.")
        if self.force_intervention_positive and not self.intervention_field:
            raise ValueError(
                "'acp.intervention_field' must be set when 'acp.force_intervention_positive=true'."
            )


@dataclass
class ValueTrainConfig:
    device: str | None = None
    use_amp: bool = False
    batch_size: int = 32
    num_workers: int = 0
    max_steps: int = 10_000
    log_freq: int = 20
    save_every: int = 500

    def validate(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("'train.batch_size' must be > 0.")
        if self.num_workers < 0:
            raise ValueError("'train.num_workers' must be >= 0.")
        if self.max_steps <= 0:
            raise ValueError("'train.max_steps' must be > 0.")
        if self.log_freq <= 0:
            raise ValueError("'train.log_freq' must be > 0.")
        if self.save_every <= 0:
            raise ValueError("'train.save_every' must be > 0.")


@dataclass
class ValueTrainPipelineConfig:
    dataset: ValueDatasetConfig
    value: ValueModelConfig = field(default_factory=SiglipGemmaValueConfig)
    targets: ValueTargetsConfig = field(default_factory=ValueTargetsConfig)
    acp: ValueAdvantageConfig = field(default_factory=ValueAdvantageConfig)
    train: ValueTrainConfig = field(default_factory=ValueTrainConfig)
    output_dir: Path | None = None
    job_name: str | None = None
    seed: int | None = 42
    push_to_hub: bool = False
    repo_id: str | None = None
    wandb: WandBConfig = field(default_factory=WandBConfig)
    use_value_training_preset: bool = True
    optimizer: OptimizerConfig | None = None
    scheduler: LRSchedulerConfig | None = None

    def validate(self) -> None:
        self.value.validate()
        self.targets.validate()
        self.acp.validate()
        self.train.validate()

        if not self.dataset.repo_id:
            raise ValueError("'dataset.repo_id' must be provided.")
        normalized_default = normalize_episode_success_label(self.dataset.default_success)
        if normalized_default is None:
            raise ValueError("'dataset.default_success' must be either 'success' or 'failure'.")
        self.dataset.default_success = normalized_default
        if not self.dataset.success_field:
            raise ValueError("'dataset.success_field' must be non-empty.")
        if self.push_to_hub and not self.repo_id:
            raise ValueError("'repo_id' argument missing. Please specify it to push the model to the hub.")

        if self.use_value_training_preset:
            self.optimizer = self.value.get_optimizer_preset()
            self.scheduler = self.value.get_scheduler_preset()
        elif self.optimizer is None:
            raise ValueError("'optimizer' must be provided when 'use_value_training_preset=false'.")

        if not self.job_name:
            dataset_id = self.dataset.repo_id.replace("/", "_")
            self.job_name = f"{dataset_id}_{self.value.type}_value"

        if self.output_dir is None:
            now = dt.datetime.now()
            train_dir = f"{now:%Y-%m-%d}/{now:%H-%M-%S}_{self.job_name}"
            self.output_dir = Path("outputs/train") / train_dir

    def to_dict(self) -> dict[str, Any]:
        return draccus.encode(self)  # type: ignore[no-any-return]
