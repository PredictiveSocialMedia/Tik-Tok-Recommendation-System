"""RL-based content optimization policy (REINFORCE, pure numpy).

Addresses the creator-side problem: given a video's features, which content
strategy should the creator optimize for to maximize predicted engagement?

The existing policy.py handles *recommendation reranking* (what to show
viewers). This module handles *content optimization* (what creators should
post) — a separate, complementary concern.

MDP formulation
---------------
State   : 12-dim feature vector extracted from a video row
          (views, engagement rate, trajectory regimes, hashtag count,
           caption length, hour-of-day cyclical encoding, engagement flag)
Action  : one of 4 discrete content strategies
          REACH      – trending/viral format for maximum reach
          ENGAGEMENT – community-focused for deep engagement rate
          CONVERSION – niche/specific for audience conversion
          RETENTION  – evergreen storytelling for watch-time
Reward  : weighted combination of views signal + engagement-rate signal +
          trajectory regime signal; weights differ per action so each
          strategy is rewarded for its own success criteria

Algorithm
---------
REINFORCE (Monte-Carlo policy gradient) with exponential moving-average
baseline subtraction and He-initialized 2-layer MLP policy network.
Entire implementation is pure numpy — no torch, peft, or trl required,
so all tests run without any ML dependencies.

Offline RL setup
----------------
Historical scraped rows are used as fixed episodes. Each row's observed
engagement (views × er × trajectory) is the reward signal. The policy
learns to assign the strategy that best predicted a video's actual success.
"""
from __future__ import annotations

import enum
import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Action space
# ---------------------------------------------------------------------------

class ContentAction(enum.IntEnum):
    REACH      = 0  # Maximize reach via trending/viral content
    ENGAGEMENT = 1  # Maximize engagement rate via community content
    CONVERSION = 2  # Maximize conversions via niche/specific content
    RETENTION  = 3  # Maximize watch-time via evergreen storytelling

N_ACTIONS = len(ContentAction)

# Per-action weights for the reward components.
# Each row sums to 1.0 so reward ∈ [0, 1].
_ACTION_WEIGHTS: Dict[int, Dict[str, float]] = {
    ContentAction.REACH:      {"views": 0.65, "er": 0.25, "spike":    0.10, "balanced": 0.00, "durable": 0.00},
    ContentAction.ENGAGEMENT: {"views": 0.40, "er": 0.45, "spike":    0.00, "balanced": 0.15, "durable": 0.00},
    ContentAction.CONVERSION: {"views": 0.25, "er": 0.55, "spike":    0.00, "balanced": 0.00, "durable": 0.20},
    ContentAction.RETENTION:  {"views": 0.35, "er": 0.55, "spike":    0.00, "balanced": 0.10, "durable": 0.00},
}

_ACTION_ADVICE: Dict[int, str] = {
    ContentAction.REACH:      (
        "Use trending hashtags (#fyp, #viral, #trending), post during peak hours "
        "(7-9 PM), open with a strong hook in the first 2 seconds."
    ),
    ContentAction.ENGAGEMENT: (
        "Use community-specific hashtags, ask a question in the caption, "
        "reply to comments early to boost algorithm placement."
    ),
    ContentAction.CONVERSION: (
        "Use niche hashtags for your exact audience, include a clear call-to-action, "
        "keep the video focused on a single specific topic."
    ),
    ContentAction.RETENTION:  (
        "Tell a complete story with a satisfying payoff, use chapter-style structure, "
        "add subtitles and on-screen text so it works on silent autoplay."
    ),
}

STATE_DIM = 12


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_state(row: Dict[str, Any]) -> np.ndarray:
    """Convert a video row dict into a 12-dim float32 state vector.

    Works with both scraped rows (plays/likes fields) and internal rows
    (engagement_metrics dict). All features are normalised to [0, 1]
    except the hour-of-day cyclical pair which lives in [-1, 1].
    """
    em = row.get("engagement_metrics") or {}
    traj = (row.get("trajectory_trace") or {})
    probs = traj.get("regime_probabilities") or {}

    # Engagement signals
    views = float(em.get("views") or row.get("plays") or 0.0)
    likes = float(em.get("likes") or row.get("likes") or 0.0)
    er = float(em.get("engagement_rate") or 0.0)
    if er == 0.0 and views > 0:
        # Approximate from raw counts when pre-computed ER is absent
        comments = float(row.get("comments_count") or 0.0)
        shares = float(row.get("shares") or 0.0)
        er = (likes + comments + shares) / max(1.0, views)

    log_views_norm = min(math.log1p(views) / math.log1p(1e7), 1.0)
    er_norm = min(er / 0.10, 1.0)

    # Trajectory regime probabilities
    spike    = float(probs.get("spike")    or 0.0)
    balanced = float(probs.get("balanced") or 0.0)
    durable  = float(probs.get("durable")  or 0.0)
    confidence = float(traj.get("regime_confidence") or 0.0)

    # Content features
    hashtag_count = len(row.get("hashtags") or [])
    ht_norm = min(hashtag_count / 20.0, 1.0)
    caption_len = len(str(row.get("caption") or ""))
    cap_norm = min(caption_len / 300.0, 1.0)

    # Time-of-day cyclical encoding (hour → sin/cos)
    hour_sin, hour_cos = 0.0, 0.0
    raw_ts = row.get("posted_at") or row.get("as_of_time") or ""
    if raw_ts:
        try:
            dt = datetime.fromisoformat(str(raw_ts).replace("Z", "+00:00"))
            hour = dt.hour
            hour_sin = math.sin(2 * math.pi * hour / 24)
            hour_cos = math.cos(2 * math.pi * hour / 24)
        except (ValueError, TypeError):
            pass

    # High-engagement binary flag (ER > 5% is well above TikTok median ~2%)
    high_er = float(er > 0.05)

    return np.array(
        [
            log_views_norm, er_norm,
            spike, balanced, durable, confidence,
            ht_norm, cap_norm,
            hour_sin, hour_cos,
            float(bool(row.get("is_verified") or False)),
            high_er,
        ],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------

def compute_reward(row: Dict[str, Any], action: int) -> float:
    """Return the reward for taking `action` given the row's observed engagement.

    Each action weights the engagement signals differently so that a strategy
    is rewarded for its own success criteria, not a single universal metric.
    Reward is in [0, 1].
    """
    em = row.get("engagement_metrics") or {}
    traj_probs = (row.get("trajectory_trace") or {}).get("regime_probabilities") or {}

    views = float(em.get("views") or row.get("plays") or 0.0)
    likes = float(em.get("likes") or row.get("likes") or 0.0)
    er = float(em.get("engagement_rate") or 0.0)
    if er == 0.0 and views > 0:
        comments = float(row.get("comments_count") or 0.0)
        shares = float(row.get("shares") or 0.0)
        er = (likes + comments + shares) / max(1.0, views)

    view_signal = min(math.log1p(views) / math.log1p(1e7), 1.0)
    er_signal   = min(er / 0.10, 1.0)
    spike    = float(traj_probs.get("spike")    or 0.0)
    balanced = float(traj_probs.get("balanced") or 0.0)
    durable  = float(traj_probs.get("durable")  or 0.0)

    w = _ACTION_WEIGHTS[action]
    reward = (
        w["views"]    * view_signal +
        w["er"]       * er_signal   +
        w["spike"]    * spike        +
        w["balanced"] * balanced     +
        w["durable"]  * durable
    )
    return float(np.clip(reward, 0.0, 1.0))


def optimal_action(row: Dict[str, Any]) -> int:
    """Return the action that maximises reward for this row (oracle baseline)."""
    rewards = [compute_reward(row, a) for a in range(N_ACTIONS)]
    return int(np.argmax(rewards))


# ---------------------------------------------------------------------------
# Policy network (pure numpy 2-layer MLP)
# ---------------------------------------------------------------------------

class PolicyNetwork:
    """Tiny He-initialised MLP: state → action probabilities.

    Implements analytical backprop for REINFORCE so no autograd framework
    is required.  Weights are stored as plain float32 numpy arrays and can
    be serialised to / from a JSON-compatible dict.
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        hidden_dim: int = 64,
        n_actions: int = N_ACTIONS,
        seed: int = 42,
    ) -> None:
        rng = np.random.RandomState(seed)
        s1 = np.sqrt(2.0 / state_dim)
        s2 = np.sqrt(2.0 / hidden_dim)
        self.W1 = (rng.randn(hidden_dim, state_dim) * s1).astype(np.float32)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W2 = (rng.randn(n_actions, hidden_dim) * s2).astype(np.float32)
        self.b2 = np.zeros(n_actions, dtype=np.float32)

    # ------------------------------------------------------------------

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        x = x - x.max()
        e = np.exp(x)
        return e / (e.sum() + 1e-9)

    def forward(self, state: np.ndarray) -> np.ndarray:
        """Return action probability vector."""
        h = np.maximum(0.0, self.W1 @ state + self.b1)
        return self._softmax(self.W2 @ h + self.b2)

    def sample(self, state: np.ndarray, rng: np.random.RandomState) -> int:
        probs = self.forward(state)
        return int(rng.choice(len(probs), p=probs))

    def greedy(self, state: np.ndarray) -> int:
        return int(np.argmax(self.forward(state)))

    def log_prob(self, state: np.ndarray, action: int) -> float:
        return float(np.log(self.forward(state)[action] + 1e-8))

    def entropy(self, state: np.ndarray) -> float:
        p = self.forward(state)
        return float(-np.sum(p * np.log(p + 1e-8)))

    # ------------------------------------------------------------------
    # REINFORCE gradient step (analytical backprop through softmax + ReLU)
    # ------------------------------------------------------------------

    def update(
        self, state: np.ndarray, action: int, advantage: float, lr: float
    ) -> None:
        """Single REINFORCE gradient-ascent step.

        Gradient: ∇θ log π(a|s) × advantage
        Analytical for a 2-layer MLP with ReLU hidden and softmax output.
        """
        # Forward
        h = np.maximum(0.0, self.W1 @ state + self.b1)
        probs = self._softmax(self.W2 @ h + self.b2)

        # Gradient of log π(a|s) w.r.t. logits = e_a - π  (softmax identity)
        onehot = np.zeros_like(probs)
        onehot[action] = 1.0
        d_logits = advantage * (onehot - probs)

        # Layer 2 gradients
        dW2 = np.outer(d_logits, h)
        db2 = d_logits

        # Backprop through ReLU into layer 1
        d_h    = self.W2.T @ d_logits
        d_pre  = d_h * (h > 0)
        dW1    = np.outer(d_pre, state)
        db1    = d_pre

        # Gradient ascent
        self.W1 += lr * dW1
        self.b1 += lr * db1
        self.W2 += lr * dW2
        self.b2 += lr * db2

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "W1": self.W1.tolist(),
            "b1": self.b1.tolist(),
            "W2": self.W2.tolist(),
            "b2": self.b2.tolist(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PolicyNetwork":
        net = cls.__new__(cls)
        net.W1 = np.array(d["W1"], dtype=np.float32)
        net.b1 = np.array(d["b1"], dtype=np.float32)
        net.W2 = np.array(d["W2"], dtype=np.float32)
        net.b2 = np.array(d["b2"], dtype=np.float32)
        return net


# ---------------------------------------------------------------------------
# Config + Agent
# ---------------------------------------------------------------------------

@dataclass
class RLContentPolicyConfig:
    hidden_dim: int = 64
    learning_rate: float = 1e-3
    epochs: int = 10
    baseline_decay: float = 0.95  # EMA decay for reward baseline
    seed: int = 42


class REINFORCEAgent:
    """REINFORCE policy-gradient agent for content strategy optimisation.

    Training (offline):
      For each epoch, iterate through all rows in random order.
      Sample an action from the current policy, observe the reward,
      compute advantage = reward − EMA_baseline, back-propagate.

    Evaluation:
      Greedy action selection; reports mean_reward, improvement over
      a uniformly random baseline, and mean policy entropy.
    """

    def __init__(self, cfg: Optional[RLContentPolicyConfig] = None) -> None:
        self.cfg = cfg or RLContentPolicyConfig()
        self.policy = PolicyNetwork(hidden_dim=self.cfg.hidden_dim, seed=self.cfg.seed)
        self._rng = np.random.RandomState(self.cfg.seed)
        self._baseline = 0.0

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        train_rows: List[Dict[str, Any]],
        val_rows: Optional[List[Dict[str, Any]]] = None,
    ) -> "REINFORCEAgent":
        """Train the policy on historical rows and return self."""
        if not train_rows:
            raise ValueError("train_rows is empty — cannot train.")

        indices = np.arange(len(train_rows))
        lr = self.cfg.learning_rate

        for epoch in range(self.cfg.epochs):
            self._rng.shuffle(indices)
            epoch_rewards: List[float] = []

            for idx in indices:
                row = train_rows[int(idx)]
                state = extract_state(row)
                action = self.policy.sample(state, self._rng)
                reward = compute_reward(row, action)

                advantage = reward - self._baseline
                self._baseline = (
                    self.cfg.baseline_decay * self._baseline
                    + (1.0 - self.cfg.baseline_decay) * reward
                )
                self.policy.update(state, action, advantage, lr)
                epoch_rewards.append(reward)

            mean_r = float(np.mean(epoch_rewards))
            logger.info("Epoch %2d/%d  mean_reward=%.4f  baseline=%.4f",
                        epoch + 1, self.cfg.epochs, mean_r, self._baseline)

        if val_rows:
            metrics = self.evaluate(val_rows)
            logger.info("Validation — %s", "  ".join(f"{k}={v:.4f}" for k, v in sorted(metrics.items())))

        return self

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, rows: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate greedy policy vs. random baseline.

        Returns:
          mean_reward           – average reward under greedy policy
          mean_reward_random    – average reward under uniform random policy
          reward_vs_random      – lift over random
          mean_entropy          – mean policy entropy (higher = more exploratory)
          action_accuracy       – fraction of greedy actions matching oracle argmax
        """
        if not rows:
            return {
                "mean_reward": 0.0,
                "mean_reward_random": 0.0,
                "reward_vs_random": 0.0,
                "mean_entropy": 0.0,
                "action_accuracy": 0.0,
            }

        rewards, rand_rewards, entropies, correct = [], [], [], []
        for row in rows:
            state = extract_state(row)
            action = self.policy.greedy(state)
            rewards.append(compute_reward(row, action))
            rand_rewards.append(compute_reward(row, int(self._rng.randint(N_ACTIONS))))
            entropies.append(self.policy.entropy(state))
            correct.append(float(action == optimal_action(row)))

        return {
            "mean_reward":        float(np.mean(rewards)),
            "mean_reward_random": float(np.mean(rand_rewards)),
            "reward_vs_random":   float(np.mean(rewards)) - float(np.mean(rand_rewards)),
            "mean_entropy":       float(np.mean(entropies)),
            "action_accuracy":    float(np.mean(correct)),
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def recommend(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Return a content strategy recommendation for a single video row."""
        state = extract_state(row)
        probs = self.policy.forward(state)
        action = int(np.argmax(probs))
        action_name = ContentAction(action).name
        return {
            "action":     action_name,
            "confidence": float(probs[action]),
            "probs":      {a.name: float(probs[a]) for a in ContentAction},
            "advice":     _ACTION_ADVICE[action],
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        payload = {
            "config": {
                "hidden_dim":     self.cfg.hidden_dim,
                "learning_rate":  self.cfg.learning_rate,
                "epochs":         self.cfg.epochs,
                "baseline_decay": self.cfg.baseline_decay,
                "seed":           self.cfg.seed,
            },
            "baseline": self._baseline,
            "policy":   self.policy.to_dict(),
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=2)
        logger.info("Policy saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "REINFORCEAgent":
        with open(path) as fh:
            payload = json.load(fh)
        cfg = RLContentPolicyConfig(**payload["config"])
        agent = cls(cfg)
        agent._baseline = float(payload["baseline"])
        agent.policy = PolicyNetwork.from_dict(payload["policy"])
        logger.info("Policy loaded from %s", path)
        return agent
