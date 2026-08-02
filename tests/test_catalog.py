"""Unit tests for models.catalog."""

from __future__ import annotations

import pytest

from models.catalog import get_model
from models.ddm.ddm3 import DDM3


class TestCatalog:
    def test_get_ddm3(self):
        assert get_model("ddm3") is DDM3

    def test_unknown_slug_raises(self):
        with pytest.raises(KeyError, match="Unknown model slug"):
            get_model("definitely-not-registered-xyz")
