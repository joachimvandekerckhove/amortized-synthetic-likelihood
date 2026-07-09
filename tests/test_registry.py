"""Unit tests for asl.registry."""

from __future__ import annotations

import pytest

from asl.registry import get_model, registered_slugs


class TestRegistry:
    def test_register_and_get(self, toy_model):
        assert get_model("toy") is toy_model
        assert "toy" in registered_slugs()

    def test_unknown_slug_raises(self):
        with pytest.raises(KeyError, match="Unknown model slug"):
            get_model("definitely-not-registered-xyz")

    def test_error_message_mentions_asl(self):
        with pytest.raises(KeyError, match="ASL"):
            get_model("definitely-not-registered-xyz")
