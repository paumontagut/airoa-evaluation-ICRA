"""Smoke test for SmolVLAPolicy adapter with synthetic observations.

Usage (requires GPU + checkpoint):
    CHECKPOINT=/path/to/pretrained_model python -m pytest tests/test_smolvla_policy.py -v

If CHECKPOINT is not set, tests are skipped.
"""

import os
import numpy as np
import pytest

CHECKPOINT = os.environ.get("CHECKPOINT")


@pytest.fixture(scope="module")
def policy():
    if not CHECKPOINT:
        pytest.skip("CHECKPOINT env var not set")
    # Add src to path
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from smolvla_policy import SmolVLAPolicy
    return SmolVLAPolicy(checkpoint_dir=CHECKPOINT, device="cuda")


def make_obs(prompt="pick up the cup"):
    return {
        "head_rgb": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        "hand_rgb": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        "state": np.random.randn(8).astype(np.float32),
        "prompt": prompt,
    }


def test_infer_shape(policy):
    result = policy.infer(make_obs())
    assert "actions" in result
    actions = result["actions"]
    assert actions.dtype == np.float32
    assert actions.ndim == 2
    assert actions.shape[1] == 11  # 11-DoF HSR action
    assert actions.shape[0] > 0   # at least one timestep


def test_infer_values_reasonable(policy):
    result = policy.infer(make_obs())
    actions = result["actions"]
    # Unnormalized actions should be small deltas, not huge numbers
    assert np.all(np.abs(actions) < 10.0), f"Actions seem too large: max={np.abs(actions).max()}"


def test_reset(policy):
    policy.reset()
    result = policy.infer(make_obs())
    assert "actions" in result


def test_multiple_infer_calls(policy):
    """Verify that the action queue works across multiple calls."""
    for i in range(5):
        result = policy.infer(make_obs(f"task {i}"))
        assert result["actions"].shape[1] == 11
