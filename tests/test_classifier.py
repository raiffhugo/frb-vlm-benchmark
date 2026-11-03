from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import run_pipeline
from vlm_classifier.classifier import (
    ClassifierConfig,
    VLMClassifier,
    _is_fatal_model_error,
    read_git_provenance,
)
from vlm_classifier.models import DryRunVLM
from vlm_classifier.prompt import prompt_for_task


def _write_manifest(root: Path, *, extra: dict[str, object] | None = None) -> Path:
    (root / "image.png").write_bytes(b"image")
    manifest = root / "manifest.jsonl"
    record: dict[str, object] = {
        "sample_id": "sample_000000",
        "image_path": "image.png",
        "true_label": "FRB",
        "simulation_parameters": {},
    }
    record.update(extra or {})
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return manifest


class EchoVLM(DryRunVLM):
    """Resposta deterministica por imagem, como o Gemma com decodificacao gulosa."""

    def classify(self, *, image_path, prompt, sample_id):  # type: ignore[no-untyped-def]
        digest = hashlib.sha256(f"{sample_id}|{prompt}".encode("utf-8")).digest()
        probability = round(digest[0] / 255, 4)
        label = "FRB" if probability >= 0.5 else "NON_FRB"
        return json.dumps({"label": label, "frb_probability": probability})


def _write_many(root: Path, n: int) -> Path:
    manifest = root / "manifest.jsonl"
    lines = []
    for index in range(n):
        image = root / f"image_{index}.png"
        image.write_bytes(f"image {index}".encode("utf-8"))
        lines.append(json.dumps({
            "sample_id": f"sample_{index:06d}",
            "image_path": image.name,
            "true_label": "FRB" if index % 2 else "RFI",
            "simulation_parameters": {},
        }))
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def _classifier(root: Path, manifest: Path, output: Path, git: dict, prompt: Path | None = None) -> VLMClassifier:
    return VLMClassifier(
        config=ClassifierConfig(
            manifest_path=manifest, output_path=output, task="frb-binary", prompt_path=prompt
        ),
        model=EchoVLM(seed=1, labels=("FRB", "NON_FRB")),
        project_root=root,
        git_provenance=git,
    )


class ClassifierTests(unittest.TestCase):
    def test_explicit_git_provenance_is_frozen_before_inference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = _write_manifest(root)
            captured = {
                "commit": "commit-before-runtime-log",
                "tracked_worktree_dirty": False,
            }
            classifier = VLMClassifier(
                config=ClassifierConfig(
                    manifest_path=manifest,
                    output_path=root / "predictions.jsonl",
                    task="frb-binary",
                ),
                model=DryRunVLM(seed=42, labels=("FRB", "NON_FRB")),
                project_root=root,
                git_provenance=captured,
            )

            captured["tracked_worktree_dirty"] = True
            predictions = classifier.run(overwrite=True)

            self.assertEqual(
                predictions[0]["provenance"]["git"],
                {
                    "commit": "commit-before-runtime-log",
                    "tracked_worktree_dirty": False,
                },
            )

    def test_records_manifest_image_and_prompt_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = _write_manifest(
                root,
                extra={
                    "source_sample_id": "frb_00000",
                    "file_group_id": "file_a",
                    "window_index": 2,
                    "window_count": 6,
                    "window_start_seconds": 2.0,
                    "window_stop_seconds": 4.0,
                },
            )
            classifier = VLMClassifier(
                config=ClassifierConfig(
                    manifest_path=manifest,
                    output_path=root / "predictions.jsonl",
                    task="frb-binary",
                ),
                model=DryRunVLM(seed=42, labels=("FRB", "NON_FRB")),
                project_root=root,
                git_provenance={"commit": None, "tracked_worktree_dirty": None},
            )

            prediction = classifier.run(overwrite=True)[0]

            provenance = prediction["provenance"]
            self.assertEqual(provenance["manifest"]["path"], "manifest.jsonl")
            self.assertEqual(
                provenance["manifest"]["sha256"],
                hashlib.sha256(manifest.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                provenance["images"],
                [
                    {
                        "image_path": "image.png",
                        "sha256": hashlib.sha256(b"image").hexdigest(),
                    }
                ],
            )
            self.assertEqual(
                provenance["prompt"]["sha256"],
                hashlib.sha256(
                    prompt_for_task("frb-binary").encode("utf-8")
                ).hexdigest(),
            )
            self.assertEqual(provenance["model"]["mode"], "dry-run")
            self.assertEqual(prediction["source_sample_id"], "frb_00000")
            self.assertEqual(prediction["file_group_id"], "file_a")
            self.assertEqual(prediction["window_index"], 2)
            self.assertEqual(prediction["window_stop_seconds"], 4.0)

    def test_prompt_file_is_used_byte_for_byte_and_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = _write_manifest(root)
            prompt = root / "lean.txt"
            prompt.write_bytes(b"Classify.\nReturn JSON {\"label\": \"FRB\"}")
            seen: list[str] = []

            class RecordingVLM(DryRunVLM):
                def classify(self, *, image_path, prompt, sample_id):  # type: ignore[no-untyped-def]
                    seen.append(prompt)
                    return super().classify(image_path=image_path, prompt=prompt, sample_id=sample_id)

            classifier = VLMClassifier(
                config=ClassifierConfig(
                    manifest_path=manifest,
                    output_path=root / "predictions.jsonl",
                    task="frb-binary",
                    prompt_path=Path("lean.txt"),
                ),
                model=RecordingVLM(seed=42, labels=("FRB", "NON_FRB")),
                project_root=root,
                git_provenance={},
            )

            prediction = classifier.run(overwrite=True)[0]

            self.assertEqual(seen, [prompt.read_text(encoding="utf-8")])
            self.assertEqual(prediction["provenance"]["prompt"]["source"], "file:lean.txt")
            self.assertEqual(
                prediction["provenance"]["prompt"]["sha256"],
                hashlib.sha256(prompt.read_bytes()).hexdigest(),
            )

    def test_builtin_prompt_source_is_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            classifier = VLMClassifier(
                config=ClassifierConfig(
                    manifest_path=_write_manifest(root),
                    output_path=root / "predictions.jsonl",
                    task="frb-binary",
                ),
                model=DryRunVLM(seed=42, labels=("FRB", "NON_FRB")),
                project_root=root,
                git_provenance={},
            )

            prediction = classifier.run(overwrite=True)[0]

            self.assertEqual(prediction["provenance"]["prompt"]["source"], "builtin:frb-binary")

    def test_rejects_output_that_overwrites_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = _write_manifest(root)

            with self.assertRaises(ValueError):
                VLMClassifier(
                    config=ClassifierConfig(
                        manifest_path=manifest,
                        output_path=manifest,
                        task="frb-binary",
                    ),
                    model=DryRunVLM(seed=42, labels=("FRB", "NON_FRB")),
                    project_root=root,
                    git_provenance={},
                )

    def test_pipeline_captures_git_before_loading_model(self) -> None:
        events: list[str] = []
        captured = {
            "commit": "commit-before-model-load",
            "tracked_worktree_dirty": False,
        }
        fake_model = SimpleNamespace(model_id="fake-model")

        def read_git(_root: Path) -> dict[str, object]:
            events.append("git")
            return captured

        def load_model(**kwargs: object) -> SimpleNamespace:
            events.append("model")
            self.assertEqual(kwargs["revision"], "revision")
            return fake_model

        with (
            mock.patch(
                "run_pipeline.read_git_provenance",
                side_effect=read_git,
            ),
            mock.patch(
                "run_pipeline.Gemma4VLM",
                side_effect=load_model,
            ),
            mock.patch("run_pipeline.VLMClassifier") as classifier_type,
        ):
            classifier_type.return_value.run.return_value = []
            run_pipeline._classify(
                config=SimpleNamespace(output_dir=Path("/tmp/dataset")),
                project_root=Path("/tmp"),
                model_name="gemma4",
                model_id="fake-model",
                model_revision="revision",
                device_map="cpu",
                dtype="float32",
                max_new_tokens=32,
                attn_implementation=None,
                cache_implementation=None,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                generation_seed=None,
                dry_run=False,
                seed=42,
                max_retries=0,
                retry_delay=0.0,
                overwrite=True,
                task="frb-binary",
                decision_threshold=0.5,
            )

        self.assertEqual(events, ["git", "model"])
        self.assertIs(
            classifier_type.call_args.kwargs["git_provenance"],
            captured,
        )

    def test_git_dirty_check_ignores_only_the_runtime_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "logs").mkdir()
            (root / "logs" / "simulation.log").write_text("old\n", encoding="utf-8")
            (root / "config.yaml").write_text("seed: 42\n", encoding="utf-8")
            git = ["git", "-c", "user.name=t", "-c", "user.email=t@t"]
            for command in (["init", "-q"], ["add", "."], ["commit", "-q", "-m", "init"]):
                subprocess.run([*git, *command], cwd=root, check=True)

            (root / "logs" / "simulation.log").write_text("new\n", encoding="utf-8")
            self.assertFalse(read_git_provenance(root)["tracked_worktree_dirty"])

            (root / "config.yaml").write_text("seed: 7\n", encoding="utf-8")
            self.assertTrue(read_git_provenance(root)["tracked_worktree_dirty"])

    def test_resume_continues_a_truncated_run_with_the_same_answers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = _write_many(root, 5)
            output = root / "predictions.jsonl"
            git = {"commit": "abc", "tracked_worktree_dirty": False}

            full = _classifier(root, manifest, output, git).run(overwrite=True)
            lines = output.read_text(encoding="utf-8").splitlines(keepends=True)
            output.write_text("".join(lines[:3]) + lines[3][:40], encoding="utf-8")  # queda no meio
            resumed = _classifier(root, manifest, output, git).run(resume=True)

            self.assertEqual(
                [r["raw_model_response"] for r in resumed], [r["raw_model_response"] for r in full]
            )
            on_disk = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(on_disk), 5)
            self.assertNotIn("resume", on_disk[2]["provenance"])
            self.assertEqual(
                on_disk[3]["provenance"]["resume"], {"resumed_at_record": 3, "previous_commit": "abc"}
            )

    def test_resume_refuses_a_different_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = _write_many(root, 3)
            output = root / "predictions.jsonl"
            git = {"commit": "abc", "tracked_worktree_dirty": False}
            _classifier(root, manifest, output, git).run(overwrite=True)
            (root / "other.txt").write_text("Outro prompt.", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "prompt"):
                _classifier(root, manifest, output, git, prompt=Path("other.txt")).run(resume=True)

    def test_resume_across_commits_needs_the_same_inference_code(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "vlm_classifier").mkdir()
            for name in ("models.py", "classifier.py"):
                (root / "vlm_classifier" / name).write_text("v1\n", encoding="utf-8")
            git_cmd = ["git", "-c", "user.name=t", "-c", "user.email=t@t"]

            def commit(message: str) -> str:
                subprocess.run([*git_cmd, "add", "."], cwd=root, check=True)
                subprocess.run([*git_cmd, "commit", "-q", "-m", message], cwd=root, check=True)
                return subprocess.run(
                    ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
                ).stdout.strip()

            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            first = commit("a")
            manifest = _write_many(root, 4)
            output = root / "out" / "predictions.jsonl"
            output.parent.mkdir()
            _classifier(root, manifest, output, {"commit": first, "tracked_worktree_dirty": False}).run(
                overwrite=True
            )
            lines = output.read_text(encoding="utf-8").splitlines(keepends=True)
            output.write_text("".join(lines[:2]), encoding="utf-8")

            (root / "vlm_classifier" / "classifier.py").write_text("v2\n", encoding="utf-8")
            orchestration_only = commit("b")
            resumed = _classifier(
                root, manifest, output, {"commit": orchestration_only, "tracked_worktree_dirty": False}
            ).run(resume=True)
            self.assertEqual(len(resumed), 4)

            output.write_text("".join(lines[:2]), encoding="utf-8")
            (root / "vlm_classifier" / "models.py").write_text("v2\n", encoding="utf-8")
            inference_changed = commit("c")
            with self.assertRaisesRegex(ValueError, "inference code changed"):
                _classifier(
                    root, manifest, output, {"commit": inference_changed, "tracked_worktree_dirty": False}
                ).run(resume=True)

    def test_detects_cuda_device_side_assert_as_fatal(self) -> None:
        error = "AcceleratorError: CUDA error: device-side assert triggered"

        self.assertTrue(_is_fatal_model_error(error))

    def test_treats_parser_error_as_non_fatal(self) -> None:
        error = "ValueError: Resposta sem objeto JSON."

        self.assertFalse(_is_fatal_model_error(error))


if __name__ == "__main__":
    unittest.main()
