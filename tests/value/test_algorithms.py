#!/usr/bin/env python

import numpy as np
import pytest

from lerobot.value.algorithms import (
    EpisodeTargetInfo,
    binarize_advantages,
    compute_n_step_advantages,
    compute_normalized_rewards_from_targets,
    compute_normalized_value_targets,
    compute_task_thresholds,
)


def test_compute_normalized_value_targets_and_rewards():
    episode_indices = np.array([0, 0, 0, 1, 1], dtype=np.int64)
    frame_indices = np.array([0, 1, 2, 0, 1], dtype=np.int64)
    episode_info = {
        0: EpisodeTargetInfo(episode_index=0, task_index=0, length=3, success=True),
        1: EpisodeTargetInfo(episode_index=1, task_index=0, length=2, success=False),
    }

    targets = compute_normalized_value_targets(
        episode_indices=episode_indices,
        frame_indices=frame_indices,
        episode_info=episode_info,
        c_fail_coef=1.0,
    )
    expected_targets = np.array([-0.5, -0.25, 0.0, -1.0, -0.5], dtype=np.float32)
    assert np.allclose(targets, expected_targets)

    rewards = compute_normalized_rewards_from_targets(
        targets=targets,
        episode_indices=episode_indices,
        frame_indices=frame_indices,
    )
    expected_rewards = np.array(
        [
            expected_targets[0] - expected_targets[1],
            expected_targets[1] - expected_targets[2],
            0.0,
            -0.5,
            -0.5,
        ],
        dtype=np.float32,
    )
    assert np.allclose(rewards, expected_rewards)


def test_compute_n_step_advantages():
    rewards = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    values = np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float32)
    episode_indices = np.array([0, 0, 0, 0], dtype=np.int64)
    frame_indices = np.array([0, 1, 2, 3], dtype=np.int64)

    advantages = compute_n_step_advantages(
        rewards=rewards,
        values=values,
        episode_indices=episode_indices,
        frame_indices=frame_indices,
        n_step=2,
    )
    expected = np.array([4.0, 6.0, 5.5, 2.0], dtype=np.float32)
    assert np.allclose(advantages, expected)


def test_compute_thresholds_and_binarize_with_intervention_override():
    task_indices = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    advantages = np.array([0.1, 0.2, 0.3, -1.0, 0.0, 1.0], dtype=np.float32)
    thresholds = compute_task_thresholds(task_indices, advantages, positive_ratio=1 / 3)
    indicator = binarize_advantages(task_indices, advantages, thresholds)
    assert indicator.tolist() == [0, 0, 1, 0, 0, 1]

    interventions = np.array([0, 1, 0, 0, 0, 0], dtype=np.float32)
    indicator_with_override = binarize_advantages(
        task_indices,
        advantages,
        thresholds,
        interventions=interventions,
        force_intervention_positive=True,
    )
    assert indicator_with_override.tolist() == [0, 1, 1, 0, 0, 1]


def test_binarize_advantages_requires_interventions_when_force_enabled():
    with pytest.raises(ValueError):
        binarize_advantages(
            task_indices=np.array([0], dtype=np.int64),
            advantages=np.array([0.1], dtype=np.float32),
            thresholds={0: 0.0},
            interventions=None,
            force_intervention_positive=True,
        )
