"""Tests for CLI entry point."""
from __future__ import annotations

from mlops_orchestrator.presentation.cli.main import main


class TestCLI:
    def test_main_function_exists(self):
        """main() should be importable and callable."""
        assert callable(main)
