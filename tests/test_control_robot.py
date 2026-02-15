#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.scripts.lerobot_calibrate import CalibrateConfig, calibrate
from lerobot.scripts.lerobot_human_inloop_record import human_inloop_record
from lerobot.scripts.lerobot_record import (
    ACPInferenceConfig,
    DatasetRecordConfig,
    PolicySyncDualArmExecutor,
    RecordConfig,
    _predict_policy_action_with_acp_inference,
    record_loop,
    record,
)
from lerobot.scripts.lerobot_replay import DatasetReplayConfig, ReplayConfig, replay
from lerobot.scripts.lerobot_teleoperate import TeleoperateConfig, teleoperate
from tests.fixtures.constants import DUMMY_REPO_ID
from tests.mocks.mock_robot import MockRobot, MockRobotConfig
from tests.mocks.mock_teleop import MockTeleop, MockTeleopConfig


def test_calibrate():
    robot_cfg = MockRobotConfig()
    cfg = CalibrateConfig(robot=robot_cfg)
    calibrate(cfg)


def test_teleoperate():
    robot_cfg = MockRobotConfig()
    teleop_cfg = MockTeleopConfig()
    cfg = TeleoperateConfig(
        robot=robot_cfg,
        teleop=teleop_cfg,
        teleop_time_s=0.1,
    )
    teleoperate(cfg)


def test_record_and_resume(tmp_path):
    robot_cfg = MockRobotConfig()
    teleop_cfg = MockTeleopConfig()
    dataset_cfg = DatasetRecordConfig(
        repo_id=DUMMY_REPO_ID,
        single_task="Dummy task",
        root=tmp_path / "record",
        num_episodes=1,
        episode_time_s=0.1,
        reset_time_s=0,
        push_to_hub=False,
    )
    cfg = RecordConfig(
        robot=robot_cfg,
        dataset=dataset_cfg,
        teleop=teleop_cfg,
        play_sounds=False,
    )

    dataset = record(cfg)

    assert dataset.fps == 30
    assert dataset.meta.total_episodes == dataset.num_episodes == 1
    assert dataset.meta.total_frames == dataset.num_frames == 3
    assert dataset.meta.total_tasks == 1

    cfg.resume = True
    # Mock the revision to prevent Hub calls during resume
    with (
        patch("lerobot.datasets.lerobot_dataset.get_safe_version") as mock_get_safe_version,
        patch("lerobot.datasets.lerobot_dataset.snapshot_download") as mock_snapshot_download,
    ):
        mock_get_safe_version.return_value = "v3.0"
        mock_snapshot_download.return_value = str(tmp_path / "record")
        dataset = record(cfg)

    assert dataset.meta.total_episodes == dataset.num_episodes == 2
    assert dataset.meta.total_frames == dataset.num_frames == 6
    assert dataset.meta.total_tasks == 1


def test_record_adds_episode_success_and_collector_policy_id(tmp_path):
    robot_cfg = MockRobotConfig()
    teleop_cfg = MockTeleopConfig()
    root = tmp_path / "record_with_annotations"
    dataset_cfg = DatasetRecordConfig(
        repo_id=DUMMY_REPO_ID,
        single_task="Dummy task",
        root=root,
        num_episodes=1,
        episode_time_s=0.1,
        reset_time_s=0,
        push_to_hub=False,
    )
    cfg = RecordConfig(
        robot=robot_cfg,
        dataset=dataset_cfg,
        teleop=teleop_cfg,
        play_sounds=False,
        enable_episode_outcome_labeling=True,
        default_episode_success="failure",
        enable_collector_policy_id=True,
    )

    dataset = record(cfg)
    assert "complementary_info.collector_policy_id" in dataset.features

    reloaded = LeRobotDataset(DUMMY_REPO_ID, root=root)
    assert reloaded[0]["complementary_info.collector_policy_id"] == "human"
    assert "episode_success" in reloaded.meta.episodes.column_names
    assert reloaded.meta.episodes[0]["episode_success"] == "failure"


def test_human_inloop_record_works_without_policy_and_saves_annotations(tmp_path):
    robot_cfg = MockRobotConfig()
    teleop_cfg = MockTeleopConfig()
    root = tmp_path / "hil_no_policy"
    dataset_cfg = DatasetRecordConfig(
        repo_id=DUMMY_REPO_ID,
        single_task="Dummy task",
        root=root,
        num_episodes=1,
        episode_time_s=0.1,
        reset_time_s=0,
        push_to_hub=False,
    )
    cfg = RecordConfig(
        robot=robot_cfg,
        dataset=dataset_cfg,
        teleop=teleop_cfg,
        play_sounds=False,
    )

    dataset = human_inloop_record(cfg)
    assert cfg.intervention_state_machine_enabled is False
    assert cfg.collector_policy_id_policy == "human"
    assert "complementary_info.collector_policy_id" in dataset.features

    reloaded = LeRobotDataset(DUMMY_REPO_ID, root=root)
    assert reloaded[0]["complementary_info.collector_policy_id"] == "human"
    assert "episode_success" in reloaded.meta.episodes.column_names
    assert reloaded.meta.episodes[0]["episode_success"] == "failure"


def test_record_loop_sets_leader_manual_control_during_reset():
    class MockTeleopWithManualControl(MockTeleop):
        def __init__(self, config):
            super().__init__(config)
            self.manual_control_calls = []

        def set_manual_control(self, enabled: bool) -> None:
            self.manual_control_calls.append(enabled)

    robot = MockRobot(MockRobotConfig())
    teleop = MockTeleopWithManualControl(MockTeleopConfig())
    robot.connect()
    teleop.connect()
    try:
        record_loop(
            robot=robot,
            events={
                "exit_early": True,
                "rerecord_episode": False,
                "stop_recording": False,
                "toggle_intervention": False,
                "episode_outcome": None,
            },
            fps=30,
            teleop_action_processor=lambda x: x[0],
            robot_action_processor=lambda x: x[0],
            robot_observation_processor=lambda x: x,
            teleop=teleop,
            policy=None,
            control_time_s=0.1,
        )
    finally:
        if teleop.is_connected:
            teleop.disconnect()
        if robot.is_connected:
            robot.disconnect()

    assert teleop.manual_control_calls == [True]


def test_record_and_replay(tmp_path):
    robot_cfg = MockRobotConfig()
    teleop_cfg = MockTeleopConfig()
    record_dataset_cfg = DatasetRecordConfig(
        repo_id=DUMMY_REPO_ID,
        single_task="Dummy task",
        root=tmp_path / "record_and_replay",
        num_episodes=1,
        episode_time_s=0.1,
        push_to_hub=False,
    )
    record_cfg = RecordConfig(
        robot=robot_cfg,
        dataset=record_dataset_cfg,
        teleop=teleop_cfg,
        play_sounds=False,
    )
    replay_dataset_cfg = DatasetReplayConfig(
        repo_id=DUMMY_REPO_ID,
        episode=0,
        root=tmp_path / "record_and_replay",
    )
    replay_cfg = ReplayConfig(
        robot=robot_cfg,
        dataset=replay_dataset_cfg,
        play_sounds=False,
    )

    record(record_cfg)

    # Mock the revision to prevent Hub calls during replay
    with (
        patch("lerobot.datasets.lerobot_dataset.get_safe_version") as mock_get_safe_version,
        patch("lerobot.datasets.lerobot_dataset.snapshot_download") as mock_snapshot_download,
    ):
        mock_get_safe_version.return_value = "v3.0"
        mock_snapshot_download.return_value = str(tmp_path / "record_and_replay")
        replay(replay_cfg)


def test_policy_sync_dual_arm_executor():
    robot = MagicMock()
    robot.send_action.return_value = {"motor_1.pos": 10.0}
    teleop = MagicMock()

    executor = PolicySyncDualArmExecutor(robot=robot, teleop=teleop, parallel_dispatch=True)
    action = {"motor_1.pos": 10.0}
    sent_action = executor.send_action(action)
    executor.shutdown()

    assert sent_action == action
    robot.send_action.assert_called_once_with(action)
    teleop.send_feedback.assert_called_once_with(action)


def test_record_config_rejects_cfg_without_acp_enable():
    robot_cfg = MockRobotConfig()
    teleop_cfg = MockTeleopConfig()
    dataset_cfg = DatasetRecordConfig(
        repo_id=DUMMY_REPO_ID,
        single_task="Dummy task",
        num_episodes=1,
        episode_time_s=0.1,
        reset_time_s=0,
        push_to_hub=False,
    )

    with pytest.raises(ValueError, match="acp_inference.use_cfg=true"):
        RecordConfig(
            robot=robot_cfg,
            dataset=dataset_cfg,
            teleop=teleop_cfg,
            play_sounds=False,
            acp_inference=ACPInferenceConfig(enable=False, use_cfg=True, cfg_beta=0.6),
        )


def test_record_config_rejects_negative_cfg_beta():
    robot_cfg = MockRobotConfig()
    teleop_cfg = MockTeleopConfig()
    dataset_cfg = DatasetRecordConfig(
        repo_id=DUMMY_REPO_ID,
        single_task="Dummy task",
        num_episodes=1,
        episode_time_s=0.1,
        reset_time_s=0,
        push_to_hub=False,
    )

    with pytest.raises(ValueError, match="cfg_beta"):
        RecordConfig(
            robot=robot_cfg,
            dataset=dataset_cfg,
            teleop=teleop_cfg,
            play_sounds=False,
            acp_inference=ACPInferenceConfig(enable=True, use_cfg=False, cfg_beta=-0.1),
        )


def test_acp_inference_without_cfg_appends_positive_prompt():
    class _StaticPolicy:
        def __init__(self, value: float):
            self.value = value
            self.tasks = []

        def select_action(self, batch):
            self.tasks.append(batch["task"])
            return torch.tensor([[self.value, self.value, self.value]], dtype=torch.float32)

    observation_frame = {"observation.state": np.array([0.0, 0.0, 0.0], dtype=np.float32)}
    policy = _StaticPolicy(value=2.0)

    action = _predict_policy_action_with_acp_inference(
        observation_frame=observation_frame,
        policy=policy,
        device=torch.device("cpu"),
        preprocessor=lambda x: x,
        postprocessor=lambda x: x,
        use_amp=False,
        task="Pick and place",
        robot_type="mock_robot",
        acp_inference=ACPInferenceConfig(enable=True, use_cfg=False, cfg_beta=0.6),
    )

    assert torch.allclose(action, torch.tensor([[2.0, 2.0, 2.0]], dtype=torch.float32))
    assert policy.tasks[-1] == "Pick and place\nAdvantage: positive"


def test_acp_inference_with_cfg_blends_cond_and_uncond_actions():
    class _StaticPolicy:
        def __init__(self, value: float):
            self.value = value
            self.tasks = []

        def select_action(self, batch):
            self.tasks.append(batch["task"])
            return torch.tensor([[self.value, self.value, self.value]], dtype=torch.float32)

    observation_frame = {"observation.state": np.array([0.0, 0.0, 0.0], dtype=np.float32)}
    policy_cond = _StaticPolicy(value=3.0)
    policy_uncond = _StaticPolicy(value=1.0)

    action = _predict_policy_action_with_acp_inference(
        observation_frame=observation_frame,
        policy=policy_cond,
        device=torch.device("cpu"),
        preprocessor=lambda x: x,
        postprocessor=lambda x: x,
        use_amp=False,
        task="Pick and place",
        robot_type="mock_robot",
        acp_inference=ACPInferenceConfig(enable=True, use_cfg=True, cfg_beta=0.5),
        policy_uncond=policy_uncond,
        preprocessor_uncond=lambda x: x,
        postprocessor_uncond=lambda x: x,
    )

    assert torch.allclose(action, torch.tensor([[2.0, 2.0, 2.0]], dtype=torch.float32))
    assert policy_cond.tasks[-1] == "Pick and place\nAdvantage: positive"
    assert policy_uncond.tasks[-1] == "Pick and place"
