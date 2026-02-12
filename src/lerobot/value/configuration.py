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

import abc
from dataclasses import dataclass, field
from typing import Any

import draccus

from lerobot.optim.optimizers import AdamWConfig, OptimizerConfig
from lerobot.optim.schedulers import LRSchedulerConfig


@dataclass
class ValueModelConfig(draccus.ChoiceRegistry, abc.ABC):  # type: ignore[misc]
    num_bins: int = 201
    bin_min: float = -1.0
    bin_max: float = 0.0
    state_feature: str = "observation.state"
    task_index_feature: str = "task_index"

    @property
    def type(self) -> str:
        choice_name = self.get_choice_name(self.__class__)
        if not isinstance(choice_name, str):
            raise TypeError(f"Expected choice name as str, got {type(choice_name)}")
        return choice_name

    @property
    def backbone(self) -> str:
        return self.type

    @classmethod
    def default_choice_name(cls) -> str | None:
        return "mlp"

    def validate(self) -> None:
        if self.num_bins < 2:
            raise ValueError("'value.num_bins' must be >= 2.")
        if self.bin_min >= self.bin_max:
            raise ValueError("'value.bin_min' must be < 'value.bin_max'.")
        if not self.state_feature:
            raise ValueError("'value.state_feature' must be set.")
        if not self.task_index_feature:
            raise ValueError("'value.task_index_feature' must be set.")

    def get_optimizer_preset(self) -> OptimizerConfig:
        return AdamWConfig(lr=1e-4, weight_decay=1e-5, grad_clip_norm=10.0)

    def get_scheduler_preset(self) -> LRSchedulerConfig | None:
        return None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ValueModelConfig":
        data = dict(payload)
        model_type = str(data.pop("type", cls.default_choice_name() or "mlp"))
        config_cls = cls.get_choice_class(model_type)
        return config_cls(**data)


@ValueModelConfig.register_subclass("mlp")
@dataclass
class MLPValueConfig(ValueModelConfig):
    hidden_dims: list[int] = field(default_factory=lambda: [512, 512])
    task_embedding_dim: int = 32
    dropout: float = 0.1

    def validate(self) -> None:
        super().validate()
        if len(self.hidden_dims) == 0:
            raise ValueError("'value.hidden_dims' cannot be empty.")
        if any(dim <= 0 for dim in self.hidden_dims):
            raise ValueError("'value.hidden_dims' must contain positive integers.")
        if self.task_embedding_dim <= 0:
            raise ValueError("'value.task_embedding_dim' must be > 0.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("'value.dropout' must be within [0, 1).")
