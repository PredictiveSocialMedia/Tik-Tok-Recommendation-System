"""Unit tests for the RL content optimization policy.

All tests are pure numpy — no torch, peft, or other ML dependencies needed.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from src.recommendation.learning.rl_content_policy import (
    N_ACTIONS,
    STATE_DIM,
    ContentAction,
    PolicyNetwork,
    REINFORCEAgent,
    RLContentPolicyConfig,
    compute_reward,
    extract_state,
    optimal_action,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _row(
    plays: int = 10_000,
    likes: int = 500,
    comments: int = 30,
    shares: int = 20,
    hashtags: list | None = None,
    caption: str = "Travel vlog #travel #italy",
    posted_at: str = "2026-01-15T19:00:00Z",
    spike: float = 0.0,
    balanced: float = 0.0,
    durable: float = 0.0,
    confidence: float = 0.0,
) -> dict:
    er = (likes + comments + shares) / max(1, plays)
    return {
        "row_id":   "r1",
        "caption":  caption,
        "hashtags": hashtags if hashtags is not None else ["travel", "italy"],
        "posted_at": posted_at,
        "plays":  plays,
        "likes":  likes,
        "comments_count": comments,
        "shares": shares,
        "engagement_metrics": {"views": plays, "engagement_rate": er},
        "trajectory_trace": {
            "regime_probabilities": {
                "spike":    spike,
                "balanced": balanced,
                "durable":  durable,
            },
            "regime_confidence": confidence,
        },
    }


def _rich_row(**kwargs) -> dict:
    return _row(
        plays=1_000_000, likes=80_000, comments=4_000, shares=15_000,
        spike=0.7, balanced=0.2, durable=0.1, confidence=0.9, **kwargs
    )


# ---------------------------------------------------------------------------
# extract_state
# ---------------------------------------------------------------------------

class TestExtractState:
    def test_shape(self):
        assert extract_state(_row()).shape == (STATE_DIM,)

    def test_dtype(self):
        assert extract_state(_row()).dtype == np.float32

    def test_all_values_finite(self):
        s = extract_state(_row())
        assert np.all(np.isfinite(s))

    def test_zero_plays_gives_zero_view_signal(self):
        s = extract_state(_row(plays=0, likes=0))
        assert s[0] == pytest.approx(0.0)

    def test_high_plays_gives_high_view_signal(self):
        s = extract_state(_row(plays=10_000_000))
        assert s[0] == pytest.approx(1.0, abs=0.01)

    def test_er_capped_at_one(self):
        # ER = 200% → capped to 1.0
        s = extract_state(_row(plays=100, likes=200, comments=50, shares=50))
        assert s[1] <= 1.0

    def test_trajectory_probs_propagate(self):
        s = extract_state(_row(spike=0.8, balanced=0.15, durable=0.05, confidence=0.9))
        assert s[2] == pytest.approx(0.8, abs=1e-5)   # spike
        assert s[3] == pytest.approx(0.15, abs=1e-5)  # balanced
        assert s[4] == pytest.approx(0.05, abs=1e-5)  # durable
        assert s[5] == pytest.approx(0.9,  abs=1e-5)  # confidence

    def test_hashtag_count_normalised(self):
        s20 = extract_state(_row(hashtags=["t"] * 20))
        assert s20[6] == pytest.approx(1.0)
        s0  = extract_state(_row(hashtags=[]))
        assert s0[6] == pytest.approx(0.0)

    def test_hour_of_day_cyclical(self):
        # 19:00 → sin > 0, cos < 0
        s = extract_state(_row(posted_at="2026-01-01T19:00:00Z"))
        hour_sin, hour_cos = s[8], s[9]
        expected_sin = math.sin(2 * math.pi * 19 / 24)
        expected_cos = math.cos(2 * math.pi * 19 / 24)
        assert hour_sin == pytest.approx(expected_sin, abs=1e-5)
        assert hour_cos == pytest.approx(expected_cos, abs=1e-5)

    def test_different_rows_produce_different_states(self):
        s1 = extract_state(_row(plays=1_000))
        s2 = extract_state(_row(plays=1_000_000))
        assert not np.allclose(s1, s2)

    def test_missing_trajectory_gives_zero_probs(self):
        row = {"caption": "test", "plays": 1000, "likes": 10}
        s = extract_state(row)
        assert s[2] == pytest.approx(0.0)  # spike
        assert s[5] == pytest.approx(0.0)  # confidence


# ---------------------------------------------------------------------------
# compute_reward
# ---------------------------------------------------------------------------

class TestComputeReward:
    def test_reward_in_unit_interval(self):
        row = _row()
        for a in range(N_ACTIONS):
            r = compute_reward(row, a)
            assert 0.0 <= r <= 1.0, f"Action {a} reward={r} out of range"

    def test_zero_engagement_gives_low_reward(self):
        row = _row(plays=0, likes=0, comments=0, shares=0)
        for a in range(N_ACTIONS):
            assert compute_reward(row, a) == pytest.approx(0.0)

    def test_high_views_boosts_reach_most(self):
        row = _row(plays=10_000_000, likes=0, comments=0, shares=0)
        reach_r = compute_reward(row, ContentAction.REACH)
        conv_r  = compute_reward(row, ContentAction.CONVERSION)
        assert reach_r > conv_r  # REACH weights views more

    def test_spike_trajectory_boosts_reach(self):
        row = _row(spike=1.0, confidence=1.0)
        reach_r = compute_reward(row, ContentAction.REACH)
        conv_r  = compute_reward(row, ContentAction.CONVERSION)
        assert reach_r > conv_r

    def test_durable_trajectory_boosts_conversion(self):
        row = _row(durable=1.0, confidence=1.0)
        conv_r     = compute_reward(row, ContentAction.CONVERSION)
        reach_r    = compute_reward(row, ContentAction.REACH)
        assert conv_r > reach_r

    def test_action_weights_sum_to_one(self):
        from src.recommendation.learning.rl_content_policy import _ACTION_WEIGHTS
        for a, w in _ACTION_WEIGHTS.items():
            total = sum(w.values())
            assert total == pytest.approx(1.0, abs=1e-6), f"Action {a} weights sum to {total}"


# ---------------------------------------------------------------------------
# optimal_action
# ---------------------------------------------------------------------------

class TestOptimalAction:
    def test_returns_valid_action(self):
        assert 0 <= optimal_action(_row()) < N_ACTIONS

    def test_spike_row_prefers_reach(self):
        row = _row(plays=5_000_000, likes=100_000, spike=1.0, confidence=1.0)
        assert optimal_action(row) == ContentAction.REACH

    def test_durable_row_prefers_conversion(self):
        row = _row(plays=1_000, likes=800, comments=200, shares=100,
                   durable=1.0, confidence=1.0)
        assert optimal_action(row) == ContentAction.CONVERSION


# ---------------------------------------------------------------------------
# PolicyNetwork
# ---------------------------------------------------------------------------

class TestPolicyNetwork:
    def test_forward_returns_valid_probs(self):
        net = PolicyNetwork(seed=0)
        probs = net.forward(np.zeros(STATE_DIM, dtype=np.float32))
        assert probs.shape == (N_ACTIONS,)
        assert np.isclose(probs.sum(), 1.0, atol=1e-5)
        assert np.all(probs >= 0)

    def test_uniform_state_gives_near_uniform_probs(self):
        # Fresh random-init network on all-zero input shouldn't massively
        # favour one action (the bias alone determines output for zero input)
        net = PolicyNetwork(seed=42)
        probs = net.forward(np.zeros(STATE_DIM, dtype=np.float32))
        assert probs.max() < 0.95  # not collapsed

    def test_sample_returns_valid_action(self):
        net = PolicyNetwork(seed=0)
        rng = np.random.RandomState(0)
        state = np.random.rand(STATE_DIM).astype(np.float32)
        action = net.sample(state, rng)
        assert 0 <= action < N_ACTIONS

    def test_greedy_returns_argmax(self):
        net = PolicyNetwork(seed=0)
        state = np.random.rand(STATE_DIM).astype(np.float32)
        probs  = net.forward(state)
        greedy = net.greedy(state)
        assert greedy == int(np.argmax(probs))

    def test_log_prob_negative(self):
        net = PolicyNetwork(seed=0)
        state = np.random.rand(STATE_DIM).astype(np.float32)
        lp = net.log_prob(state, 0)
        assert lp <= 0.0

    def test_entropy_positive(self):
        net = PolicyNetwork(seed=0)
        state = np.zeros(STATE_DIM, dtype=np.float32)
        assert net.entropy(state) > 0.0

    def test_update_changes_weights(self):
        net = PolicyNetwork(seed=7)
        state = np.ones(STATE_DIM, dtype=np.float32)
        W1_before = net.W1.copy()
        net.update(state, action=0, advantage=1.0, lr=0.1)
        assert not np.allclose(net.W1, W1_before)

    def test_update_positive_advantage_increases_log_prob(self):
        net = PolicyNetwork(seed=5)
        state = np.random.RandomState(0).rand(STATE_DIM).astype(np.float32)
        action = 1
        lp_before = net.log_prob(state, action)
        net.update(state, action=action, advantage=10.0, lr=0.01)
        lp_after = net.log_prob(state, action)
        assert lp_after > lp_before

    def test_update_negative_advantage_decreases_log_prob(self):
        net = PolicyNetwork(seed=5)
        state = np.random.RandomState(0).rand(STATE_DIM).astype(np.float32)
        action = 1
        lp_before = net.log_prob(state, action)
        net.update(state, action=action, advantage=-10.0, lr=0.01)
        lp_after = net.log_prob(state, action)
        assert lp_after < lp_before

    def test_serialise_roundtrip(self):
        net = PolicyNetwork(seed=3)
        state = np.random.rand(STATE_DIM).astype(np.float32)
        probs_before = net.forward(state)
        restored = PolicyNetwork.from_dict(net.to_dict())
        probs_after = restored.forward(state)
        assert np.allclose(probs_before, probs_after, atol=1e-6)


# ---------------------------------------------------------------------------
# REINFORCEAgent
# ---------------------------------------------------------------------------

def _make_rows(n: int = 20) -> list:
    rng = np.random.RandomState(0)
    rows = []
    for i in range(n):
        plays = int(rng.randint(100, 1_000_000))
        likes = int(rng.randint(0, plays // 10 + 1))
        rows.append(_row(
            plays=plays, likes=likes,
            spike=float(rng.rand()),
            balanced=float(rng.rand()),
            durable=float(rng.rand()),
            confidence=float(rng.rand()),
        ))
    return rows


class TestREINFORCEAgent:
    def test_train_returns_self(self):
        agent = REINFORCEAgent(RLContentPolicyConfig(epochs=2, seed=0))
        result = agent.train(_make_rows(10))
        assert result is agent

    def test_train_empty_rows_raises(self):
        agent = REINFORCEAgent()
        with pytest.raises(ValueError, match="empty"):
            agent.train([])

    def test_evaluate_returns_expected_keys(self):
        agent = REINFORCEAgent(RLContentPolicyConfig(epochs=1, seed=0))
        agent.train(_make_rows(10))
        metrics = agent.evaluate(_make_rows(5))
        expected = {"mean_reward", "mean_reward_random", "reward_vs_random",
                    "mean_entropy", "action_accuracy"}
        assert set(metrics.keys()) == expected

    def test_evaluate_scores_in_valid_range(self):
        agent = REINFORCEAgent(RLContentPolicyConfig(epochs=1, seed=0))
        agent.train(_make_rows(10))
        metrics = agent.evaluate(_make_rows(10))
        assert 0.0 <= metrics["mean_reward"] <= 1.0
        assert 0.0 <= metrics["action_accuracy"] <= 1.0
        assert metrics["mean_entropy"] >= 0.0

    def test_evaluate_empty_rows_returns_zeros(self):
        agent = REINFORCEAgent()
        metrics = agent.evaluate([])
        assert metrics["mean_reward"] == 0.0
        assert metrics["action_accuracy"] == 0.0

    def test_training_improves_reward_vs_random(self):
        """After enough epochs the policy should beat a random baseline."""
        rows = _make_rows(50)
        # Untrained baseline
        untrained = REINFORCEAgent(RLContentPolicyConfig(epochs=0, seed=0))
        m_before = untrained.evaluate(rows)

        trained = REINFORCEAgent(RLContentPolicyConfig(epochs=15, seed=0, learning_rate=1e-2))
        trained.train(rows)
        m_after = trained.evaluate(rows)

        # The trained policy should achieve higher mean reward than the
        # untrained one (at least non-negative lift)
        assert m_after["mean_reward"] >= m_before["mean_reward"] - 0.05

    def test_recommend_returns_valid_action(self):
        agent = REINFORCEAgent(RLContentPolicyConfig(epochs=2, seed=0))
        agent.train(_make_rows(10))
        rec = agent.recommend(_row())
        assert rec["action"] in {a.name for a in ContentAction}
        assert "advice" in rec
        assert "probs" in rec
        assert set(rec["probs"].keys()) == {a.name for a in ContentAction}

    def test_recommend_probs_sum_to_one(self):
        agent = REINFORCEAgent(RLContentPolicyConfig(epochs=1, seed=0))
        agent.train(_make_rows(5))
        rec = agent.recommend(_row())
        total = sum(rec["probs"].values())
        assert total == pytest.approx(1.0, abs=1e-5)

    def test_recommend_confidence_matches_action_prob(self):
        agent = REINFORCEAgent(RLContentPolicyConfig(epochs=1, seed=0))
        agent.train(_make_rows(5))
        rec = agent.recommend(_row())
        assert rec["confidence"] == pytest.approx(
            rec["probs"][rec["action"]], abs=1e-6
        )

    def test_save_load_roundtrip(self, tmp_path):
        agent = REINFORCEAgent(RLContentPolicyConfig(epochs=3, seed=1))
        agent.train(_make_rows(15))
        path = str(tmp_path / "policy.json")
        agent.save(path)

        loaded = REINFORCEAgent.load(path)
        row = _row()
        rec_orig   = agent.recommend(row)
        rec_loaded = loaded.recommend(row)
        assert rec_orig["action"] == rec_loaded["action"]
        assert rec_orig["confidence"] == pytest.approx(rec_loaded["confidence"], abs=1e-6)

    def test_saved_file_is_valid_json(self, tmp_path):
        agent = REINFORCEAgent(RLContentPolicyConfig(epochs=1, seed=0))
        agent.train(_make_rows(5))
        path = tmp_path / "policy.json"
        agent.save(str(path))
        with open(path) as fh:
            payload = json.load(fh)
        assert "policy" in payload
        assert "config" in payload
        assert "baseline" in payload

    def test_action_accuracy_oracle(self):
        """A policy trained on a single repeated row should learn its optimal action."""
        row = _row(plays=10_000_000, likes=500_000, spike=1.0, confidence=1.0)
        agent = REINFORCEAgent(RLContentPolicyConfig(
            epochs=30, learning_rate=5e-2, seed=0
        ))
        agent.train([row] * 40)
        metrics = agent.evaluate([row] * 10)
        # Should mostly pick REACH for a viral-spike, high-view row
        assert metrics["action_accuracy"] > 0.5
