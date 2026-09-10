from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from lungseg.config import load_yaml, merge_configs


class ConfigTests(unittest.TestCase):
    def test_all_shipped_configs_are_non_empty_mappings(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        paths = sorted((repository / "configs").rglob("*.yaml"))

        self.assertEqual(len(paths), 6)
        for path in paths:
            with self.subTest(path=path.relative_to(repository)):
                self.assertTrue(load_yaml(path))

    def test_load_yaml_requires_mapping_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.yaml"
            path.write_text("- item\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "root must be a mapping"):
                load_yaml(path)

    def test_recursive_merge_does_not_mutate_inputs(self) -> None:
        base = {"model": {"channels": [16, 32], "norm": "INSTANCE"}}
        override = {"model": {"channels": [32, 64]}, "seed": 43}

        merged = merge_configs(base, override)
        merged["model"]["channels"].append(128)

        self.assertEqual(base["model"]["channels"], [16, 32])
        self.assertEqual(override["model"]["channels"], [32, 64])
        self.assertEqual(merged["model"]["norm"], "INSTANCE")
        self.assertEqual(merged["seed"], 43)


if __name__ == "__main__":
    unittest.main()
