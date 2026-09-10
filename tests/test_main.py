from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class MainCommandTests(unittest.TestCase):
    def run_main(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(ROOT / "main.py"), *arguments],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_help_lists_management_commands(self) -> None:
        result = self.run_main("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        for command in ("status", "inspect", "split", "train", "infer", "evaluate", "test"):
            self.assertIn(command, result.stdout)

    def test_status_runs_without_dataset(self) -> None:
        result = self.run_main("status")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Training readiness:", result.stdout)

    def test_train_rejects_unset_audit_parameters(self) -> None:
        result = self.run_main("train")
        self.assertEqual(result.returncode, 2)
        self.assertIn("parameters are unset", result.stderr)


if __name__ == "__main__":
    unittest.main()