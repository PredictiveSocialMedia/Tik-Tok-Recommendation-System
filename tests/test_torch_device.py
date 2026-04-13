"""Tests for shared PyTorch device selection."""

from __future__ import annotations

from src.common.torch_device import preferred_torch_device, torch_empty_cache


def test_preferred_torch_device_returns_known_string():
    dev = preferred_torch_device()
    assert dev in ("cuda", "mps", "cpu")


def test_torch_empty_cache_runs():
    torch_empty_cache()
