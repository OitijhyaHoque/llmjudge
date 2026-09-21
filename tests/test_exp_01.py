import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from exp_scripts import exp_01


class ExperimentOneTests(unittest.TestCase):
    def test_processes_every_csv_into_its_own_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = root / "inputs"
            inputs.mkdir()
            (inputs / "input-2.csv").touch()
            (inputs / "input-1.csv").touch()

            with (
                patch.object(exp_01, "INPUTS", inputs),
                patch.object(exp_01, "RESULTS", root / "results"),
                patch.object(exp_01, "judge", return_value=0) as judge,
            ):
                code = exp_01.main(["--dry-run", "--run-id", "test-run"])

        self.assertEqual(code, 0)
        self.assertEqual(
            [Path(call.kwargs["items"]).name for call in judge.call_args_list],
            ["input-1.csv", "input-2.csv"],
        )
        self.assertEqual(
            [Path(call.kwargs["out"]).parts[-2:] for call in judge.call_args_list],
            [("test-run", "input-1"), ("test-run", "input-2")],
        )


if __name__ == "__main__":
    unittest.main()
