from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from revision_v4.common import MODEL_REVISIONS
from revision_v4.gpu_runs import MODELS, RUNS, classify_command, is_complete, output_path, select_runs


class GpuRunsTests(unittest.TestCase):
    def test_command_repeats_paper_flags_with_pinned_revision_and_frozen_prompt(self) -> None:
        run = select_runs("e1", ["binary_lean"])[0]

        command = classify_command(run, "e4b")

        joined = " ".join(command)
        for flag in (
            "--device-map cuda",
            "--dtype auto",
            "--max-new-tokens 1024",
            "--cache-implementation static",
            "--generation-seed 42",
            "--task frb-binary",
            "--prompt-file prompts_v4/binary_lean.txt",
            "--manifest dataset_e1_binary/metadata/image_manifest.jsonl",
            "--output results_v4/e1/e4b_binary_lean/predictions.jsonl",
        ):
            self.assertIn(flag, joined)
        self.assertIn(
            f"--model-revision {MODEL_REVISIONS['google/gemma-4-E4B-it']}", joined
        )

    def test_every_run_uses_a_registered_prompt(self) -> None:
        for runs in RUNS.values():
            for run in runs:
                for model in MODELS:
                    self.assertTrue(run.prompt_for(model).exists(), run.prompt_for(model))

    def test_e7_filler_prompt_belongs_to_each_model(self) -> None:
        run = select_runs("e7", ["v4_filler"])[0]

        for model in MODELS:
            command = " ".join(classify_command(run, model))
            self.assertIn(f"--prompt-file prompts_v4/e7_v4_filler_{model}.txt", command)
            self.assertIn(f"--output results_v4/e7/{model}_v4_filler/predictions.jsonl", command)
            self.assertIn("--manifest results_v4/selections/e7_heldout400.jsonl", command)

    def test_multiclass_runs_on_all_3000_e1_samples(self) -> None:
        run = select_runs("e1", ["multiclass"])[0]

        self.assertEqual(run.task, "multiclass")
        self.assertEqual(run.manifest, Path("dataset_e1/metadata/image_manifest.jsonl"))

    def test_complete_only_when_output_matches_manifest_length(self) -> None:
        run = select_runs("e1", ["binary_full"])[0]
        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                run.manifest.parent.mkdir(parents=True)
                run.manifest.write_text("{}\n{}\n", encoding="utf-8")
                output = output_path(run, "e2b")
                self.assertFalse(is_complete(run, "e2b"))
                output.parent.mkdir(parents=True)
                output.write_text("{}\n", encoding="utf-8")
                self.assertFalse(is_complete(run, "e2b"))
                output.write_text("{}\n{}\n", encoding="utf-8")
                self.assertTrue(is_complete(run, "e2b"))
            finally:
                os.chdir(cwd)


if __name__ == "__main__":
    unittest.main()
