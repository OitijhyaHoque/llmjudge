"""llmjudge/colab.py off Colab: the Drive mirror, and where the endpoint comes from.

Nothing here mounts Drive or starts a runtime. `MyDrive` is a temporary directory, which
is all the mirror ever sees, and the parts that only exist inside Colab -- `drive.mount`
and `userdata` -- are absent, which is exactly the case the module is written to survive.

    python3 -m unittest tests.test_colab           (from the repository root)
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from llmjudge import colab  # noqa: E402


def write(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def read(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


class Paths(unittest.TestCase):
    def test_drive_prefix_becomes_mydrive(self):
        self.assertEqual(colab.resolve("drive:genmd-judge/results/pilot"),
                         os.path.join(colab.MYDRIVE, "genmd-judge/results/pilot"))
        self.assertEqual(colab.resolve("drive:/genmd-judge"),
                         os.path.join(colab.MYDRIVE, "genmd-judge"))

    def test_any_other_path_is_left_alone(self):
        self.assertEqual(colab.resolve("/tmp/x"), "/tmp/x")
        self.assertEqual(colab.resolve("out/pilot"), os.path.abspath("out/pilot"))

    def test_mount_off_colab_says_so_instead_of_raising(self):
        self.assertFalse(colab.mount(os.path.join(tempfile.mkdtemp(), "nope")))


class Mirror(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.local = os.path.join(self.tmp, "local")
        self.drive = os.path.join(self.tmp, "drive")
        os.makedirs(self.local)
        os.makedirs(self.drive)

    def test_jsonl_is_copied_by_its_tail_only(self):
        results = os.path.join(self.local, "results.jsonl")
        write(results, '{"id": "a"}\n')
        self.assertEqual(colab.sync(self.local, self.drive), [])
        self.assertEqual(read(os.path.join(self.drive, "results.jsonl")), '{"id": "a"}\n')

        # The run appends. Only the new bytes cross, and the copy on Drive still reads
        # as the whole file: this is what keeps a 50,000-row file from being re-uploaded
        # every hundred seconds.
        with open(results, "a", encoding="utf-8") as f:
            f.write('{"id": "b"}\n')
        self.assertEqual(colab.sync(self.local, self.drive), [])
        self.assertEqual(read(os.path.join(self.drive, "results.jsonl")),
                         '{"id": "a"}\n{"id": "b"}\n')
        self.assertEqual(colab.lines(os.path.join(self.drive, "results.jsonl")), 2)

    def test_other_files_are_copied_whole_including_the_prompt(self):
        write(os.path.join(self.local, "summary.json"), '{"rows": 1}')
        write(os.path.join(self.local, "prompt", "system.md"), "judge it\n")
        self.assertEqual(colab.sync(self.local, self.drive), [])
        self.assertEqual(read(os.path.join(self.drive, "summary.json")), '{"rows": 1}')
        # The prompt sits in a subdirectory, and a results directory on Drive that does
        # not carry the prompt that produced it is not much use six months later.
        self.assertEqual(read(os.path.join(self.drive, "prompt", "system.md")), "judge it\n")

    def test_the_pid_of_a_dead_runtime_is_not_mirrored(self):
        write(os.path.join(self.local, "judge.pid"), "123\n")
        colab.sync(self.local, self.drive)
        self.assertFalse(os.path.exists(os.path.join(self.drive, "judge.pid")))

    def test_restore_brings_the_last_session_down_but_never_overwrites(self):
        write(os.path.join(self.drive, "results.jsonl"), '{"id": "a"}\n')
        write(os.path.join(self.drive, "prompt", "system.md"), "judge it\n")
        write(os.path.join(self.local, "run.json"), "mine\n")
        write(os.path.join(self.drive, "run.json"), "theirs\n")
        self.assertEqual(colab.restore(self.drive, self.local), 2)
        self.assertEqual(read(os.path.join(self.local, "results.jsonl")), '{"id": "a"}\n')
        self.assertEqual(read(os.path.join(self.local, "prompt", "system.md")), "judge it\n")
        # A file that is already local belongs to this session and is ahead of Drive.
        self.assertEqual(read(os.path.join(self.local, "run.json")), "mine\n")

    def test_a_drive_that_is_ahead_is_left_alone_rather_than_corrupted(self):
        # A local file shorter than the copy on Drive means the restore did not happen.
        # Appending its tail would interleave two runs, so nothing is copied and the
        # line counts disagree -- which is what run() reports as an incomplete mirror.
        write(os.path.join(self.drive, "results.jsonl"), '{"id": "a"}\n{"id": "b"}\n')
        write(os.path.join(self.local, "results.jsonl"), '{"id": "c"}\n')
        self.assertEqual(colab.sync(self.local, self.drive), [])
        self.assertEqual(colab.lines(os.path.join(self.drive, "results.jsonl")), 2)
        self.assertEqual(colab.lines(os.path.join(self.local, "results.jsonl")), 1)

    def test_an_unwritable_drive_is_reported_and_not_raised(self):
        write(os.path.join(self.local, "summary.json"), "{}")
        os.chmod(self.drive, 0o500)
        try:
            problems = colab.sync(self.local, self.drive)
        finally:
            os.chmod(self.drive, 0o700)
        self.assertEqual([name for name, _ in problems], ["summary.json"])

    def test_lines_of_a_file_that_is_not_there(self):
        self.assertEqual(colab.lines(os.path.join(self.drive, "results.jsonl")), 0)


class Endpoint(unittest.TestCase):
    def setUp(self):
        self.saved = {k: os.environ.pop(k, None)
                      for k in colab.ENV_BASE_URL + colab.ENV_API_KEY + colab.ENV_MODEL}

    def tearDown(self):
        for k, v in self.saved.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v

    def test_the_serving_cell_is_enough(self):
        os.environ["LLMJUDGE_BASE_URL"] = "http://127.0.0.1:8000/v1"
        os.environ["LLMJUDGE_API_KEY"] = "secret"
        os.environ["LLMJUDGE_MODEL"] = "medgemma-27b-it"
        options: dict = {}
        colab.endpoint(options)
        self.assertEqual(options["base_url"], "http://127.0.0.1:8000/v1")
        self.assertEqual(options["model"], "medgemma-27b-it")
        self.assertEqual(options["api_key"], "secret")
        # Nothing cuts a slow reply in front of localhost, so the 95 s default, which
        # exists because of Cloudflare, would only lose rows the model was still writing.
        self.assertEqual(options["timeout"], 600)

    def test_the_older_medgemma_names_still_work_and_keep_the_tunnel_timeout(self):
        os.environ["MEDGEMMA_BASE_URL"] = "https://x.trycloudflare.com/v1"
        os.environ["MEDGEMMA_MODEL"] = "medgemma-27b-it"
        options: dict = {}
        colab.endpoint(options)
        self.assertEqual(options["base_url"], "https://x.trycloudflare.com/v1")
        self.assertNotIn("timeout", options)
        self.assertNotIn("api_key", options)

    def test_what_the_cell_passed_wins(self):
        os.environ["LLMJUDGE_BASE_URL"] = "http://127.0.0.1:8000/v1"
        os.environ["LLMJUDGE_MODEL"] = "medgemma-27b-it"
        options = {"base_url": "https://api.vendor.com/v1", "model": "gpt-4o-mini",
                   "timeout": 30}
        colab.endpoint(options)
        self.assertEqual(options["base_url"], "https://api.vendor.com/v1")
        self.assertEqual(options["model"], "gpt-4o-mini")
        self.assertEqual(options["timeout"], 30)

    def test_an_endpoints_file_is_left_alone(self):
        options = {"endpoints": "configs/endpoints.toml"}
        colab.endpoint(options)
        self.assertEqual(options, {"endpoints": "configs/endpoints.toml"})

    def test_no_endpoint_is_refused_here_and_not_a_minute_later(self):
        with self.assertRaises(RuntimeError) as e:
            colab.endpoint({})
        self.assertIn("serving cell", str(e.exception))

    def test_a_url_with_no_model_is_refused_too(self):
        os.environ["LLMJUDGE_BASE_URL"] = "http://127.0.0.1:8000/v1"
        with self.assertRaises(RuntimeError):
            colab.endpoint({})


class Run(unittest.TestCase):
    def test_drive_out_off_colab_is_refused_rather_than_written_somewhere_else(self):
        with self.assertRaises(RuntimeError) as e:
            colab.run(items="/nonexistent/items.jsonl", out="drive:x", run_tag="t")
        self.assertIn("not mounted", str(e.exception))

    def test_a_missing_items_file_is_refused_before_anything_is_created(self):
        tmp = tempfile.mkdtemp()
        with self.assertRaises(FileNotFoundError):
            colab.run(items=os.path.join(tmp, "nope.jsonl"),
                      out=os.path.join(tmp, "out"), run_tag="t")


if __name__ == "__main__":
    unittest.main()
