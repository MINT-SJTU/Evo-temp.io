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

"""Export per-episode videos with value curve and ACP text overlays."""

import argparse
import concurrent.futures
import logging
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm.auto import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import encode_video_frames

DEFAULT_VALUE_FIELD = "complementary_info.value"
DEFAULT_ADVANTAGE_FIELD = "complementary_info.advantage"
DEFAULT_INDICATOR_FIELD = "complementary_info.acp_indicator"
FIXED_Y_MIN = -1.0
FIXED_Y_MAX = 0.0

_WORKER_CONTEXT: dict[str, Any] | None = None


def _parse_episodes_arg(episodes_arg: str, total_episodes: int) -> list[int]:
    value = episodes_arg.strip().lower()
    if value == "all":
        return list(range(total_episodes))

    parsed: set[int] = set()
    for token in episodes_arg.split(","):
        part = token.strip()
        if not part:
            continue
        if "-" in part:
            start_str, end_str = part.split("-", maxsplit=1)
            start = int(start_str)
            end = int(end_str)
            if end < start:
                raise ValueError(f"Invalid episode range '{part}'.")
            parsed.update(range(start, end + 1))
        else:
            parsed.add(int(part))

    episodes = sorted(parsed)
    for ep in episodes:
        if ep < 0 or ep >= total_episodes:
            raise ValueError(f"Episode index out of range: {ep}, total_episodes={total_episodes}.")
    return episodes


def _to_1d_float(values: list | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.reshape(arr.shape[0], -1)[:, 0]
    return arr.reshape(-1)


def _to_1d_int(values: list | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.int64)
    if arr.ndim > 1:
        arr = arr.reshape(arr.shape[0], -1)[:, 0]
    return arr.reshape(-1)


def _resolve_video_key(dataset: LeRobotDataset, requested_video_key: str | None) -> str:
    if len(dataset.meta.camera_keys) == 0:
        raise ValueError("No camera key found in dataset.")

    video_key = requested_video_key or dataset.meta.camera_keys[0]
    if video_key not in dataset.meta.camera_keys:
        raise ValueError(f"Unknown video_key '{video_key}'. Available camera keys: {dataset.meta.camera_keys}")
    return video_key


def _load_annotation_arrays(
    dataset: LeRobotDataset,
    value_field: str,
    advantage_field: str,
    indicator_field: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw_dataset = dataset.hf_dataset.with_format(None)
    required_fields = [value_field, advantage_field, indicator_field]
    missing_fields = [field for field in required_fields if field not in raw_dataset.column_names]
    if missing_fields:
        raise KeyError(f"Missing fields in dataset: {missing_fields}")

    values_all = _to_1d_float(raw_dataset[value_field])
    advantages_all = _to_1d_float(raw_dataset[advantage_field])
    indicators_all = _to_1d_int(raw_dataset[indicator_field])

    n_frames = len(raw_dataset)
    if len(values_all) != n_frames or len(advantages_all) != n_frames or len(indicators_all) != n_frames:
        raise RuntimeError("Overlay fields length mismatch with dataset frame count.")

    return values_all, advantages_all, indicators_all


def _build_output_path(output_dir: Path, repo_id: str, video_key: str, episode_index: int) -> Path:
    repo_tag = repo_id.replace("/", "_")
    key_tag = video_key.replace(".", "_")
    return output_dir / f"{repo_tag}_episode_{episode_index:04d}_{key_tag}.mp4"


def _load_font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    font_candidates = [
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/local/share/fonts/DejaVuSans.ttf"),
    ]
    for font_path in font_candidates:
        if font_path.exists():
            return ImageFont.truetype(str(font_path), size=size)
    return ImageFont.load_default()


def _to_pil_image(frame: torch.Tensor | np.ndarray) -> Image.Image:
    if isinstance(frame, torch.Tensor):
        frame_np = frame.detach().cpu().numpy()
        if frame_np.ndim == 3 and frame_np.shape[0] in (1, 3, 4):
            frame_np = np.transpose(frame_np, (1, 2, 0))
    else:
        frame_np = np.asarray(frame)

    if frame_np.ndim != 3:
        raise ValueError(f"Expected image with 3 dimensions, got shape={frame_np.shape}.")

    if frame_np.dtype != np.uint8:
        if frame_np.max() <= 1.5:
            frame_np = np.clip(frame_np, 0.0, 1.0) * 255.0
        else:
            frame_np = np.clip(frame_np, 0.0, 255.0)
        frame_np = frame_np.astype(np.uint8)

    if frame_np.shape[2] == 1:
        frame_np = np.repeat(frame_np, 3, axis=2)
    return Image.fromarray(frame_np, mode="RGB")


def _curve_points(
    values: np.ndarray,
    current_step: int,
    x0: int,
    y0: int,
    width: int,
    height: int,
    y_min: float,
    y_max: float,
) -> list[tuple[int, int]]:
    n = len(values)
    if n == 0:
        return []

    denom_x = max(1, n - 1)
    denom_y = max(1e-6, y_max - y_min)

    points = []
    last_step = min(current_step, n - 1)
    for i in range(last_step + 1):
        x = int(round(x0 + width * (i / denom_x)))
        y_norm = np.clip((float(values[i]) - y_min) / denom_y, 0.0, 1.0)
        y = int(round(y0 + (1.0 - y_norm) * height))
        points.append((x, y))
    return points


def _draw_overlay(
    frame: Image.Image,
    values: np.ndarray,
    current_step: int,
    advantage_t: float,
    acp_t: int,
    y_min: float,
    y_max: float,
) -> Image.Image:
    rgba = frame.convert("RGBA")
    overlay = Image.new("RGBA", rgba.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    width, height = rgba.size
    margin = max(10, width // 80)
    chart_h = max(72, height // 4)
    chart_w = width - 2 * margin
    chart_x0 = margin
    chart_y0 = height - margin - chart_h

    draw.rectangle(
        (chart_x0 - 4, chart_y0 - 4, chart_x0 + chart_w + 4, chart_y0 + chart_h + 4),
        fill=(0, 0, 0, 110),
    )
    draw.line((chart_x0, chart_y0, chart_x0, chart_y0 + chart_h), fill=(200, 200, 200, 160), width=1)
    draw.line(
        (chart_x0, chart_y0 + chart_h, chart_x0 + chart_w, chart_y0 + chart_h),
        fill=(200, 200, 200, 160),
        width=1,
    )

    mid_y = chart_y0 + chart_h // 2
    draw.line((chart_x0, mid_y, chart_x0 + chart_w, mid_y), fill=(120, 120, 120, 120), width=1)

    points = _curve_points(values, current_step, chart_x0, chart_y0, chart_w, chart_h, y_min, y_max)
    curve_width = max(2, width // 320)
    if len(points) >= 2:
        draw.line(points, fill=(255, 255, 255, 255), width=curve_width)
    if len(points) == 1:
        x, y = points[0]
        draw.ellipse((x - curve_width, y - curve_width, x + curve_width, y + curve_width), fill=(255, 255, 255, 255))
    elif len(points) > 1:
        x, y = points[-1]
        radius = max(2, curve_width + 1)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(255, 255, 255, 255))

    font_size = max(18, height // 26)
    font = _load_font(font_size)
    lines = [f"advantage: {advantage_t:+.4f}", f"acp_indicator: {int(acp_t)}"]
    line_sizes = [draw.textbbox((0, 0), text, font=font) for text in lines]
    text_w = max(box[2] - box[0] for box in line_sizes)
    text_h = sum(box[3] - box[1] for box in line_sizes) + max(4, font_size // 4)
    box_pad = max(8, font_size // 3)
    box_x1 = width - margin
    box_x0 = box_x1 - text_w - 2 * box_pad
    box_y0 = margin
    box_y1 = box_y0 + text_h + 2 * box_pad
    draw.rectangle((box_x0, box_y0, box_x1, box_y1), fill=(0, 0, 0, 150))

    text_y = box_y0 + box_pad
    for idx, text in enumerate(lines):
        box = line_sizes[idx]
        line_h = box[3] - box[1]
        draw.text((box_x0 + box_pad, text_y), text, fill=(255, 255, 255, 255), font=font)
        text_y += line_h + max(4, font_size // 4)

    return Image.alpha_composite(rgba, overlay).convert("RGB")


def export_episode_overlay_video(
    *,
    dataset: LeRobotDataset,
    episode_index: int,
    video_key: str,
    values_all: np.ndarray,
    advantages_all: np.ndarray,
    indicators_all: np.ndarray,
    output_video_path: Path,
    vcodec: str,
    overwrite: bool,
) -> None:
    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    if output_video_path.exists() and not overwrite:
        logging.info("Skip existing file: %s", output_video_path)
        return

    ep_start = int(dataset.meta.episodes["dataset_from_index"][episode_index])
    ep_end = int(dataset.meta.episodes["dataset_to_index"][episode_index])
    ep_values = values_all[ep_start:ep_end]
    ep_advantages = advantages_all[ep_start:ep_end]
    ep_indicators = indicators_all[ep_start:ep_end]

    if len(ep_values) == 0:
        logging.warning("Episode %d has zero frame, skip.", episode_index)
        return

    with tempfile.TemporaryDirectory(
        dir=output_video_path.parent, prefix=f"ep-{episode_index:04d}-frames-"
    ) as temp_dir:
        temp_path = Path(temp_dir)
        for frame_in_ep, abs_idx in enumerate(range(ep_start, ep_end)):
            item = dataset[abs_idx]
            frame = _to_pil_image(item[video_key])
            composed = _draw_overlay(
                frame=frame,
                values=ep_values,
                current_step=frame_in_ep,
                advantage_t=float(ep_advantages[frame_in_ep]),
                acp_t=int(ep_indicators[frame_in_ep]),
                y_min=FIXED_Y_MIN,
                y_max=FIXED_Y_MAX,
            )
            composed.save(temp_path / f"frame-{frame_in_ep:06d}.png")

        encode_video_frames(
            imgs_dir=temp_path,
            video_path=output_video_path,
            fps=int(dataset.fps),
            vcodec=vcodec,
            overwrite=True,
        )


def _init_worker_context(
    repo_id: str,
    root: str | None,
    revision: str | None,
    download_videos: bool,
    video_key: str,
    value_field: str,
    advantage_field: str,
    indicator_field: str,
    output_dir: str,
    vcodec: str,
    overwrite: bool,
) -> None:
    global _WORKER_CONTEXT

    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=(Path(root) if root is not None else None),
        revision=revision,
        download_videos=download_videos,
    )
    values_all, advantages_all, indicators_all = _load_annotation_arrays(
        dataset=dataset,
        value_field=value_field,
        advantage_field=advantage_field,
        indicator_field=indicator_field,
    )
    _WORKER_CONTEXT = {
        "dataset": dataset,
        "video_key": video_key,
        "values_all": values_all,
        "advantages_all": advantages_all,
        "indicators_all": indicators_all,
        "output_dir": Path(output_dir),
        "repo_id": repo_id,
        "vcodec": vcodec,
        "overwrite": overwrite,
    }


def _export_episode_in_worker(episode_index: int) -> tuple[int, str]:
    if _WORKER_CONTEXT is None:
        raise RuntimeError("Worker context was not initialized.")

    output_path = _build_output_path(
        output_dir=_WORKER_CONTEXT["output_dir"],
        repo_id=_WORKER_CONTEXT["repo_id"],
        video_key=_WORKER_CONTEXT["video_key"],
        episode_index=episode_index,
    )
    export_episode_overlay_video(
        dataset=_WORKER_CONTEXT["dataset"],
        episode_index=episode_index,
        video_key=_WORKER_CONTEXT["video_key"],
        values_all=_WORKER_CONTEXT["values_all"],
        advantages_all=_WORKER_CONTEXT["advantages_all"],
        indicators_all=_WORKER_CONTEXT["indicators_all"],
        output_video_path=output_path,
        vcodec=_WORKER_CONTEXT["vcodec"],
        overwrite=_WORKER_CONTEXT["overwrite"],
    )
    return episode_index, str(output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export per-episode videos with dynamic value curve and ACP text overlays."
    )
    parser.add_argument("--repo-id", type=str, required=True, help="Dataset repo id.")
    parser.add_argument("--root", type=Path, default=None, help="Local dataset root.")
    parser.add_argument("--revision", type=str, default=None, help="Dataset revision.")
    parser.add_argument(
        "--episodes",
        type=str,
        default="all",
        help="Episode selection, e.g. 'all', '0,1,5', or '3-10'.",
    )
    parser.add_argument(
        "--video-key",
        type=str,
        default=None,
        help="Camera key to render. Default: first available camera key.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/value_vis"),
        help="Directory where per-episode mp4 files are exported.",
    )
    parser.add_argument("--download-videos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--num-workers", type=int, default=1, help="Parallel process count for episode export.")
    parser.add_argument("--vcodec", type=str, default="h264", choices=["h264", "hevc", "libsvtav1"])
    parser.add_argument("--value-field", type=str, default=DEFAULT_VALUE_FIELD)
    parser.add_argument("--advantage-field", type=str, default=DEFAULT_ADVANTAGE_FIELD)
    parser.add_argument("--indicator-field", type=str, default=DEFAULT_INDICATOR_FIELD)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.num_workers <= 0:
        raise ValueError("--num-workers must be >= 1.")

    dataset = LeRobotDataset(
        repo_id=args.repo_id,
        root=args.root,
        revision=args.revision,
        download_videos=args.download_videos,
    )
    video_key = _resolve_video_key(dataset, args.video_key)
    episodes = _parse_episodes_arg(args.episodes, dataset.meta.total_episodes)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.num_workers == 1:
        values_all, advantages_all, indicators_all = _load_annotation_arrays(
            dataset=dataset,
            value_field=args.value_field,
            advantage_field=args.advantage_field,
            indicator_field=args.indicator_field,
        )
        for ep_idx in tqdm(episodes, desc="Exporting episodes"):
            output_path = _build_output_path(
                output_dir=args.output_dir,
                repo_id=args.repo_id,
                video_key=video_key,
                episode_index=ep_idx,
            )
            export_episode_overlay_video(
                dataset=dataset,
                episode_index=ep_idx,
                video_key=video_key,
                values_all=values_all,
                advantages_all=advantages_all,
                indicators_all=indicators_all,
                output_video_path=output_path,
                vcodec=args.vcodec,
                overwrite=args.overwrite,
            )
            logging.info("Exported episode=%d to %s", ep_idx, output_path)
        return

    dataset = None
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.num_workers,
        initializer=_init_worker_context,
        initargs=(
            args.repo_id,
            (str(args.root) if args.root is not None else None),
            args.revision,
            args.download_videos,
            video_key,
            args.value_field,
            args.advantage_field,
            args.indicator_field,
            str(args.output_dir),
            args.vcodec,
            args.overwrite,
        ),
    ) as executor:
        futures = [executor.submit(_export_episode_in_worker, ep_idx) for ep_idx in episodes]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Exporting episodes"):
            ep_idx, output_path = future.result()
            logging.info("Exported episode=%d to %s", ep_idx, output_path)


if __name__ == "__main__":
    main()
