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

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from lerobot.utils.constants import OBS_IMAGES
from lerobot.utils.import_utils import _transformers_available
from lerobot.value.configuration import SiglipGemmaValueConfig

if TYPE_CHECKING or _transformers_available:
    from transformers import AutoImageProcessor, AutoTokenizer
else:
    AutoImageProcessor = None
    AutoTokenizer = None


class SiglipGemmaValuePreprocessor:
    def __init__(self, cfg: SiglipGemmaValueConfig, dataset_camera_features: Sequence[str]):
        if AutoTokenizer is None or AutoImageProcessor is None:
            raise ImportError("transformers is not installed. Install with `pip install 'lerobot[pi0]'`.")

        self.cfg = cfg
        dataset_cameras = [key for key in dataset_camera_features if key.startswith(OBS_IMAGES)]
        if cfg.camera_features:
            unknown = sorted(set(cfg.camera_features) - set(dataset_cameras))
            if unknown:
                raise KeyError(
                    "Configured camera features are not available in dataset: "
                    f"{unknown}. Available cameras: {dataset_cameras}"
                )
            self.camera_features = list(cfg.camera_features)
        else:
            self.camera_features = list(dataset_cameras)

        if not self.camera_features:
            raise ValueError(
                "No camera features found for SiglipGemmaValueConfig. "
                "Set `value.camera_features` or ensure dataset contains `observation.images.*` columns."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.language_repo_id,
            revision=cfg.language_revision,
            use_fast=True,
        )
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token is None:
                raise ValueError(
                    f"Tokenizer for language model '{cfg.language_repo_id}' does not define pad/eos token "
                    "required for padded batches."
                )
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.image_processor = AutoImageProcessor.from_pretrained(
            cfg.vision_repo_id,
            revision=cfg.vision_revision,
            use_fast=True,
        )

    def _build_prompts(self, tasks: Sequence[str]) -> list[str]:
        prompts: list[str] = []
        for task in tasks:
            cleaned_task = str(task).strip().replace("_", " ").replace("\n", " ").strip()
            prompts.append(cleaned_task)
        return prompts

    @staticmethod
    def _to_bchw(img_batch: Tensor) -> Tensor:
        if img_batch.ndim != 4:
            raise ValueError(f"Expected image batch rank 4, got shape {tuple(img_batch.shape)}.")

        if img_batch.shape[1] in {1, 3}:  # [B,C,H,W]
            return img_batch
        if img_batch.shape[-1] in {1, 3}:  # [B,H,W,C]
            return img_batch.permute(0, 3, 1, 2)
        raise ValueError(
            "Camera tensor must be channels-first or channels-last. "
            f"Got camera batch with shape={tuple(img_batch.shape)}."
        )

    def _process_camera_batch(self, img_batch: Tensor, device: torch.device) -> Tensor:
        img_batch = self._to_bchw(img_batch).detach().to(dtype=torch.float32, device="cpu")
        do_rescale = not bool(torch.max(img_batch) <= 1.0 and torch.min(img_batch) >= 0.0)
        image_list = [img_batch[i] for i in range(img_batch.shape[0])]

        processor_kwargs: dict[str, Any] = {"return_tensors": "pt", "do_rescale": do_rescale}
        processed = self.image_processor(images=image_list, **processor_kwargs)
        pixel_values = processed["pixel_values"]
        if not isinstance(pixel_values, torch.Tensor):
            raise TypeError("Image processor did not return tensor 'pixel_values'.")
        return pixel_values.to(device=device, dtype=torch.float32, non_blocking=True)

    def _prepare_images(self, raw_batch: dict[str, Any], device: torch.device) -> tuple[Tensor, Tensor]:
        present_img_keys = [key for key in self.camera_features if key in raw_batch]
        if len(present_img_keys) == 0:
            raise ValueError(
                "All configured cameras are missing in the input batch. "
                f"expected={self.camera_features} batch_keys={list(raw_batch.keys())}"
            )

        reference_img = self._process_camera_batch(torch.as_tensor(raw_batch[present_img_keys[0]]), device=device)
        bsize = reference_img.shape[0]
        image_tensors: list[Tensor] = []
        image_masks: list[Tensor] = []

        for key in self.camera_features:
            if key in raw_batch:
                img = self._process_camera_batch(torch.as_tensor(raw_batch[key]), device=device)
                if img.shape[0] != bsize:
                    raise ValueError(
                        f"Mismatched batch size across cameras. Camera '{key}' has {img.shape[0]}, expected {bsize}."
                    )
                image_tensors.append(img)
                image_masks.append(torch.ones(bsize, dtype=torch.bool, device=device))
            else:
                image_tensors.append(torch.zeros_like(reference_img))
                image_masks.append(torch.zeros(bsize, dtype=torch.bool, device=device))

        images = torch.stack(image_tensors, dim=1)
        masks = torch.stack(image_masks, dim=1)
        return images, masks

    def __call__(self, raw_batch: dict[str, Any], device: torch.device) -> dict[str, Tensor]:
        if self.cfg.task_field not in raw_batch:
            raise KeyError(f"Missing task field '{self.cfg.task_field}' in input batch.")

        tasks_raw = raw_batch[self.cfg.task_field]
        if isinstance(tasks_raw, str) or not isinstance(tasks_raw, Sequence):
            raise TypeError(
                f"Expected task field '{self.cfg.task_field}' as a sequence of strings, got {type(tasks_raw)}."
            )
        tasks = [str(task) for task in tasks_raw]
        images, image_attention_mask = self._prepare_images(raw_batch, device)
        if len(tasks) != images.shape[0]:
            raise ValueError(f"Task count ({len(tasks)}) does not match image batch size ({images.shape[0]}).")

        prompts = self._build_prompts(tasks)
        tokenized = self.tokenizer(
            prompts,
            max_length=self.cfg.tokenizer_max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = tokenized["input_ids"].to(device=device, dtype=torch.long, non_blocking=True)
        attention_mask = tokenized["attention_mask"].to(device=device, dtype=torch.bool, non_blocking=True)

        return {
            "images": images,
            "image_attention_mask": image_attention_mask,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
