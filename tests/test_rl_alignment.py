"""Tests for RL alignment extras (RLOO/ReMax/DrGRPO/DAPO/PPO/PRM)."""
import numpy as np
import pytest

import oxideon as ox
from oxideon import (
    DAPOLoss,
    DrGRPOLoss,
    PPOClipLoss,
    PRMLoss,
    ProcessRewardHead,
    ReMaxLoss,
    RLOOLoss,
    aggregate_process_scores,
)


def t(arr):
    return ox.from_numpy(np.asarray(arr, dtype=np.float32))


def test_top_level_exports():
    assert ox.RLOOLoss is RLOOLoss
    assert ox.DAPOLoss is DAPOLoss
    assert ox.PRMLoss is PRMLoss


def test_rloo_hand_computed():
    loss, metrics = RLOOLoss()(t([[-1.0, -2.0, -3.0]]), [[1.0, 0.0, 0.0]])
    assert loss.to_numpy().tolist() == pytest.approx([-0.5])
    assert metrics["mean_reward"] == pytest.approx(1 / 3)
    loss, _ = RLOOLoss(normalize_advantages=True)(
        t([[-1.0, -2.0, -3.0]]), [[1.0, 0.0, 0.0]]
    )
    assert loss.to_numpy().tolist() == pytest.approx([-0.7071068], abs=1e-6)
    with pytest.raises(ValueError):
        RLOOLoss()(t([[0.0]]), [[1.0]])
    with pytest.raises(ValueError):
        RLOOLoss()(t([[-1.0, -2.0]]), [[1.0]])
    with pytest.raises(ValueError):
        RLOOLoss()(t([[-1.0, -2.0]]), [[1.0, float("nan")]])


def test_remax_hand_computed():
    loss, metrics = ReMaxLoss()(t([-1.0, -2.0]), [1.0, 0.0], [0.5, 0.5])
    assert loss.to_numpy().tolist() == pytest.approx([-0.25])
    assert metrics["mean_advantage"] == pytest.approx(0.0)
    with pytest.raises(ValueError):
        ReMaxLoss()(t([-1.0]), [1.0, 0.0], [0.0, 0.0])


def test_drgrpo_hand_computed_and_flat_group():
    loss, _ = DrGRPOLoss()(t([[-1.0, -2.0, -3.0]]), [[1.0, 0.0, 0.0]])
    assert loss.to_numpy().tolist() == pytest.approx([-1 / 3])
    loss, metrics = DrGRPOLoss()(t([[-1.0, -2.0]]), [[1.0, 1.0]])
    assert loss.to_numpy().tolist() == [0.0]
    assert metrics["frac_zero_advantage"] == 1.0
    with pytest.raises(ValueError):
        DrGRPOLoss()(t([[-1.0]]), [[1.0, 2.0]])


def test_dapo_clip_higher_and_dynamic_sampling():
    dapo = DAPOLoss()
    loss, metrics = dapo(t([[-0.1, -1.0]]), t([[-1.0, -1.0]]), [[2.0, 0.0]])
    assert loss.to_numpy().tolist() == pytest.approx([-0.14])
    assert metrics["frac_filtered"] == 0.0
    assert metrics["clipfrac_high"] == 0.5
    loss, metrics = dapo(t([[-1.0, -1.0]]), t([[-1.0, -1.0]]), [[1.0, 1.0]])
    assert loss.to_numpy().tolist() == [0.0]
    assert metrics["frac_filtered"] == 1.0


def test_dapo_overlong_shaping_unflattens_groups():
    dapo = DAPOLoss()
    _, plain = dapo(t([[-1.0, -1.0]]), t([[-1.0, -1.0]]), [[1.0, 1.0]])
    assert plain["frac_filtered"] == 1.0
    _, shaped = dapo(
        t([[-1.0, -1.0]]), t([[-1.0, -1.0]]), [[1.0, 1.0]],
        lengths=[[5, 100]], max_len=10,
    )
    assert shaped["frac_filtered"] == 0.0
    with pytest.raises(ValueError):
        dapo(t([[-1.0]]), t([[-1.0, -1.0]]), [[1.0]])
    with pytest.raises(ValueError):
        dapo(t([[-1.0]]), t([[-1.0]]), [[1.0]], lengths=[[3]])
    with pytest.raises(ValueError):
        DAPOLoss(clip_low=-0.1)


def test_ppo_clip_hand_computed():
    loss, metrics = PPOClipLoss(clip=0.2, vf_coef=0.0)(
        t([-1.0, -2.0]), t([-1.0, -2.0]), [1.0, -1.0], [1.0, 0.0], [0.5, 0.5]
    )
    assert loss.to_numpy().tolist() == pytest.approx([0.0])
    loss, metrics = PPOClipLoss(clip=0.2, vf_coef=0.5)(
        t([-1.0, -2.0]), t([-1.0, -2.0]), [1.0, -1.0], [1.0, 0.0], [0.5, 0.5]
    )
    assert loss.to_numpy().tolist() == pytest.approx([0.125])
    assert metrics["explained_variance"] == pytest.approx(0.0)
    loss, metrics = PPOClipLoss(clip=0.2)(
        t([0.0]), t([-1.0]), [2.0], [0.0], [0.0]
    )
    assert loss.to_numpy().tolist() == pytest.approx([-2.4])
    assert metrics["clipfrac"] == 1.0
    with pytest.raises(ValueError):
        PPOClipLoss()(t([-1.0]), t([-1.0, -2.0]), [0.0], [0.0], [0.0])


def test_process_reward_head_and_aggregation():
    head = ProcessRewardHead(8)
    scores = head(ox.from_numpy(np.random.default_rng(0).normal(size=(2, 5, 8)).astype(np.float32)))
    assert list(scores.shape) == [2, 5]
    grid = t([[0.0, 1.0, 2.0], [3.0, 1.0, 0.0]])
    assert aggregate_process_scores(grid, method="mean").to_numpy().tolist() == pytest.approx([1.0, 4 / 3])
    assert aggregate_process_scores(grid, method="min").to_numpy().tolist() == [0.0, 0.0]
    assert aggregate_process_scores(grid, method="last").to_numpy().tolist() == [2.0, 0.0]
    assert aggregate_process_scores(grid, method="prod").to_numpy().tolist() == pytest.approx(
        [0.322, 0.348], abs=1e-3
    )
    mask = [[1, 1, 0], [1, 0, 0]]
    assert aggregate_process_scores(grid, mask=mask).to_numpy().tolist() == [0.5, 3.0]
    assert aggregate_process_scores(grid, mask=mask, method="last").to_numpy().tolist() == [1.0, 3.0]
    with pytest.raises(ValueError):
        aggregate_process_scores(grid, method="median")
    with pytest.raises(ValueError):
        aggregate_process_scores(grid, mask=[[1, 0]])
    with pytest.raises(ValueError):
        aggregate_process_scores(grid, mask=[[0, 0, 0], [1, 1, 1]])
    with pytest.raises(ValueError):
        aggregate_process_scores(t([1.0, 2.0]))


def test_prm_loss():
    loss, metrics = PRMLoss()(t([[2.0, -2.0]]), [[1, 0]])
    assert loss.to_numpy().tolist() == pytest.approx([0.1269], abs=1e-4)
    assert metrics["step_accuracy"] == 1.0
    loss, _ = PRMLoss()(t([[2.0, -2.0]]), [[1, 0]], mask=[[1, 0]])
    assert loss.to_numpy().tolist() == pytest.approx([0.1269], abs=1e-4)
    with pytest.raises(ValueError):
        PRMLoss()(t([[2.0]]), [[0.5]])
    with pytest.raises(ValueError):
        PRMLoss()(t([[2.0]]), [[1]], mask=[[0]])
    with pytest.raises(ValueError):
        PRMLoss()(t([[2.0, 1.0]]), [[1]])
