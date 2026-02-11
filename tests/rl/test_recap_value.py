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

import torch

from lerobot.rl.recap_value import (
    build_episode_value_info,
    build_normalized_returns,
    expected_value_from_logits,
    project_values_to_bins,
    soft_cross_entropy,
)


class _DummyEpisodes:
    def __init__(self, columns):
        self._columns = columns
        self.column_names = list(columns)

    def __getitem__(self, key):
        return self._columns[key]


def test_build_episode_value_info_and_task_lmax():
    episodes = _DummyEpisodes(
        {
            "episode_index": [0, 1, 2],
            "length": [5, 8, 6],
            "tasks": [["task_a"], ["task_b"], ["task_a"]],
            "episode_success": ["success", "failure", "success"],
        }
    )

    info, task_lmax = build_episode_value_info(episodes)
    assert info[0].is_success is True
    assert info[1].is_success is False
    assert info[2].task == "task_a"
    assert task_lmax["task_a"] == 6
    assert task_lmax["task_b"] == 8


def test_build_normalized_returns_matches_reward_spec():
    episodes = _DummyEpisodes(
        {
            "episode_index": [0, 1],
            "length": [6, 6],
            "tasks": [["task"], ["task"]],
            "episode_success": ["success", "failure"],
        }
    )
    info, task_lmax = build_episode_value_info(episodes)

    frame_indices = torch.tensor([0, 5, 0, 5], dtype=torch.long)
    episode_indices = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    values = build_normalized_returns(
        frame_indices=frame_indices,
        episode_indices=episode_indices,
        episode_info=info,
        task_lmax=task_lmax,
        c_fail=10.0,
    )

    # success episode: G=[-5, ..., 0], failure episode: G=[-15, ..., -10], denom=(6+10)=16
    expected = torch.tensor([-5 / 16, 0.0, -15 / 16, -10 / 16], dtype=torch.float32)
    assert torch.allclose(values, expected, atol=1e-6)


def test_project_values_to_bins_soft_targets_and_loss():
    values = torch.tensor([-1.0, -0.6, 0.0], dtype=torch.float32)
    targets, bins = project_values_to_bins(values=values, num_bins=5)

    assert targets.shape == (3, 5)
    assert torch.allclose(targets.sum(dim=1), torch.ones(3), atol=1e-6)
    # -1 and 0 should be one-hot at boundaries.
    assert torch.argmax(targets[0]).item() == 0
    assert torch.argmax(targets[2]).item() == 4

    # CE should be finite and positive with arbitrary logits.
    logits = torch.tensor(
        [
            [2.0, 0.1, -0.5, -1.0, -2.0],
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [-2.0, -1.0, 0.0, 1.0, 2.0],
        ],
        dtype=torch.float32,
    )
    loss = soft_cross_entropy(logits, targets)
    assert torch.isfinite(loss)
    assert loss.item() > 0.0

    value_hat = expected_value_from_logits(logits, bins)
    assert value_hat.shape == (3,)
    assert torch.all(value_hat <= 0.0)
    assert torch.all(value_hat >= -1.0)
