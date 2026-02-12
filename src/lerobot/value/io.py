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

import datasets
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from lerobot.datasets.utils import (
    embed_images,
    get_hf_features_from_features,
    load_info,
    load_stats,
    write_info,
    write_stats,
)
from lerobot.value.configuration import ValueModelConfig
from lerobot.value.modeling import make_value_model


def get_step_dir(checkpoint_root: Path, step: int, total_steps: int) -> Path:
    digits = max(6, len(str(total_steps)))
    return checkpoint_root / f"step_{step:0{digits}d}"


def update_last_checkpoint_pointer(checkpoint_root: Path, step_dir: Path) -> None:
    last_path = checkpoint_root / "last"
    if last_path.is_symlink() or last_path.is_file():
        last_path.unlink()
    try:
        last_path.symlink_to(step_dir.name)
    except OSError:
        last_path.write_text(step_dir.name, encoding="utf-8")


def _resolve_checkpoint_ref(checkpoint_root: Path, checkpoint_ref: str | Path) -> Path:
    ref_path = Path(checkpoint_ref)
    if ref_path.is_absolute():
        candidate = ref_path
    else:
        candidate = checkpoint_root / ref_path

    if candidate.is_symlink():
        return candidate.resolve()
    if candidate.is_dir():
        return candidate
    if candidate.is_file():
        pointed = candidate.read_text(encoding="utf-8").strip()
        if not pointed:
            raise ValueError(f"Checkpoint pointer file is empty: {candidate}")
        pointed_path = checkpoint_root / pointed
        if pointed_path.exists():
            return pointed_path
    raise FileNotFoundError(f"Could not resolve checkpoint reference: {checkpoint_ref}")


def save_value_checkpoint(
    *,
    checkpoint_root: Path,
    step: int,
    total_steps: int,
    model: torch.nn.Module,
    value_config_payload: dict[str, Any],
    training_meta: dict[str, Any],
) -> Path:
    step_dir = get_step_dir(checkpoint_root, step, total_steps)
    step_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), step_dir / "weights.pt")
    (step_dir / "value_config.json").write_text(
        json.dumps(value_config_payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    (step_dir / "training_meta.json").write_text(
        json.dumps(training_meta, indent=2, sort_keys=True), encoding="utf-8"
    )
    update_last_checkpoint_pointer(checkpoint_root, step_dir)
    return step_dir


def load_value_model_from_checkpoint(
    *, checkpoint_root: Path, checkpoint_ref: str | Path, device: torch.device
) -> tuple[torch.nn.Module, ValueModelConfig, dict[str, Any]]:
    step_dir = _resolve_checkpoint_ref(checkpoint_root, checkpoint_ref)
    value_config = json.loads((step_dir / "value_config.json").read_text(encoding="utf-8"))
    model_cfg = ValueModelConfig.from_dict(value_config["model_config"])
    state_dim = int(value_config["state_dim"])
    num_tasks = int(value_config["num_tasks"])

    model = make_value_model(model_cfg, state_dim=state_dim, num_tasks=num_tasks)
    weights = torch.load(step_dir / "weights.pt", map_location=device)
    model.load_state_dict(weights)
    model.to(device)
    model.eval()
    return model, model_cfg, value_config


def _write_parquet(df: pd.DataFrame, fpath: Path, features: dict[str, dict]) -> None:
    hf_features = get_hf_features_from_features(features)
    ds = datasets.Dataset.from_dict(df.to_dict(orient="list"), features=hf_features, split="train")
    has_images = any(ft["dtype"] == "image" for ft in features.values())
    if has_images:
        ds = embed_images(ds)
    table = ds.with_format("arrow")[:]
    writer = pq.ParquetWriter(fpath, schema=table.schema, compression="snappy", use_dictionary=True)
    writer.write_table(table)
    writer.close()


def _compute_scalar_stats(values: np.ndarray) -> dict[str, list[float]]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError("Cannot compute stats for an empty array.")
    return {
        "min": [float(np.min(arr))],
        "max": [float(np.max(arr))],
        "mean": [float(np.mean(arr))],
        "std": [float(np.std(arr))],
        "count": [int(arr.size)],
        "q01": [float(np.quantile(arr, 0.01))],
        "q10": [float(np.quantile(arr, 0.10))],
        "q50": [float(np.quantile(arr, 0.50))],
        "q90": [float(np.quantile(arr, 0.90))],
        "q99": [float(np.quantile(arr, 0.99))],
    }


def write_annotations_in_place(
    *,
    dataset_root: Path,
    frame_indices: np.ndarray,
    columns: dict[str, np.ndarray],
    feature_infos: dict[str, dict[str, Any]],
) -> dict[str, str]:
    info = load_info(dataset_root)
    features = dict(info["features"])
    existing_features = set(features.keys())
    write_modes = {key: ("overwrite" if key in existing_features else "new") for key in columns}

    for key, ft in feature_infos.items():
        features[key] = dict(ft)
    info["features"] = features
    write_info(info, dataset_root)

    stats = load_stats(dataset_root) or {}
    for key, values in columns.items():
        stats[key] = _compute_scalar_stats(values)
    write_stats(stats, dataset_root)

    max_index = int(np.max(frame_indices))
    lookup_tables: dict[str, np.ndarray] = {}
    for key, values in columns.items():
        lookup_dtype = np.int64 if np.issubdtype(values.dtype, np.integer) else np.float32
        table = np.zeros(max_index + 1, dtype=lookup_dtype)
        table[frame_indices] = values.astype(lookup_dtype, copy=False)
        lookup_tables[key] = table

    parquet_files = sorted((dataset_root / "data").glob("*/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No data parquet files found under {dataset_root / 'data'}.")

    for parquet_file in parquet_files:
        df = pd.read_parquet(parquet_file)
        abs_idx = df["index"].to_numpy(dtype=np.int64)
        for key, table in lookup_tables.items():
            df[key] = table[abs_idx]
        _write_parquet(df, parquet_file, features)

    logging.info("Dataset updated in place: %s", ", ".join(columns.keys()))
    return write_modes
