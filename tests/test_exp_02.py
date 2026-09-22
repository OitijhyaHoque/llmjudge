import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from exp_scripts import exp_02
from llmjudge.run import load_prompt
from llmjudge.template import render_user


class ExperimentTwoTests(unittest.TestCase):
    def run_main(self, argv, inputs=("input-2.csv", "input-1.csv")):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root / "inputs"
            directory.mkdir()
            for name in inputs:
                (directory / name).touch()

            with (
                patch.object(exp_02, "INPUTS", directory),
                patch.object(exp_02, "RESULTS", root / "results"),
                patch.object(exp_02, "judge", return_value=0) as judge,
            ):
                return exp_02.main(argv), judge

    def test_processes_every_csv_into_its_own_directory(self):
        code, judge = self.run_main(["--dry-run", "--run-id", "test-run"])

        self.assertEqual(code, 0)
        self.assertEqual(
            [Path(call.kwargs["items"]).name for call in judge.call_args_list],
            ["input-1.csv", "input-2.csv"],
        )
        self.assertEqual(
            [Path(call.kwargs["out"]).parts[-2:] for call in judge.call_args_list],
            [("test-run", "input-1"), ("test-run", "input-2")],
        )

    def test_skips_the_pool_manifest_beside_a_table(self):
        _, judge = self.run_main(["--dry-run", "--run-id", "r"],
                                 inputs=("t.csv", "t_pool.csv"))

        self.assertEqual(
            [Path(call.kwargs["items"]).name for call in judge.call_args_list], ["t.csv"])

    def test_calls_evidencemd_unguided_and_capped(self):
        # The three EvidenceMD differences and the spend cap: each one fails silently
        # (a refused run, or a bill) rather than loudly if it is dropped.
        _, judge = self.run_main(["--dry-run", "--run-id", "r", "--limit", "10"])
        kwargs = judge.call_args_list[0].kwargs

        self.assertEqual(kwargs["base_url"], "https://evidencemd.ai/api/v1")
        self.assertEqual(kwargs["auth_header"], "x-api-key")
        self.assertTrue(kwargs["skip_model_check"])
        self.assertEqual(kwargs["guided"], "off")
        # No response_format: the vendor's json_object mode returns {"answer": "<the
        # whole answer, as a string>"}, which carries no `verdict` key to find.
        self.assertNotIn("response_format", kwargs["extra"])
        self.assertEqual(kwargs["extra"]["specialty"], "Clinical data auditor")
        self.assertEqual(kwargs["prompt"], "c3")
        self.assertEqual(kwargs["limit"], 10)
        self.assertEqual(kwargs["max_requests"], 12)


class MergedPromptTests(unittest.TestCase):
    def test_task_is_in_the_user_message_and_the_contract_in_the_system_one(self):
        prompt = load_prompt("c3-evidencemd")

        # The answer contract: read from the system message, so it has to stay there.
        self.assertEqual(prompt["order"], ("short_reason", "verdict"))
        self.assertEqual(prompt["label"], "verdict")
        self.assertEqual(prompt["values"], ["consistent", "inconsistent", "unsure"])
        # Everything the model needs to do the task travels in the one user message.
        self.assertNotIn("Decision rule", prompt["system"])
        for part in ("Decision rule", "Dataset conventions", "do not flag", "{diag_1}"):
            self.assertIn(part, prompt["user"])

    def test_renders_against_a_row(self):
        row = {c: "x" for c in re.findall(r"\{([^{}\s]+)\}", load_prompt("c3-evidencemd")["user"])}
        self.assertIn("gender: x", render_user(load_prompt("c3-evidencemd")["user"], row))


if __name__ == "__main__":
    unittest.main()
