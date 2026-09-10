from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from lungseg.utils.environment import collect_environment
from lungseg.utils.io import write_json


class EnvironmentTests(unittest.TestCase):
    def test_report_has_required_sections(self) -> None:
        report = collect_environment(Path(__file__).resolve().parents[1])

        self.assertIn("version", report["python"])
        self.assertIn("torch", report["packages"])
        self.assertIn("monai", report["packages"])
        self.assertIn("commit", report["git"])
        self.assertIn("devices", report["cuda"])

    def test_json_writer_creates_parent_and_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "nested" / "report.json"
            written = write_json(destination, {"seed": 43})

            self.assertEqual(written, destination.resolve())
            self.assertEqual(destination.read_text(encoding="utf-8"), '{\n  "seed": 43\n}\n')


if __name__ == "__main__":
    unittest.main()
