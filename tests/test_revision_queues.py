from __future__ import annotations

import unittest
from collections import Counter

from revision_v4.gpu_runs import RUNS
from revision_v4.queues import QUEUES, command


class QueueTests(unittest.TestCase):
    def test_every_vlm_run_is_queued_exactly_once_per_model(self) -> None:
        queued: Counter[tuple[str, str, str]] = Counter()
        for steps in QUEUES.values():
            for step in steps:
                if step.kind != "vlm":
                    continue
                arms = step.arms or tuple(run.arm for run in RUNS[step.experiment])
                for arm in arms:
                    queued[(step.experiment, step.model, arm)] += 1

        expected = {
            (experiment, model, run.arm)
            for experiment, runs in RUNS.items()
            for run in runs
            for model in ("e2b", "e4b")
        }
        self.assertEqual(set(queued), expected)
        self.assertEqual(set(queued.values()), {1})

    def test_vlm_steps_call_the_runner_on_their_gpu(self) -> None:
        argv, cwd, log = command(QUEUES[1][1], 1)

        self.assertIn("revision_v4.gpu_runs", argv)
        self.assertEqual(argv[argv.index("--gpu") + 1], "1")
        self.assertIn("real1600_r", argv)
        self.assertIsNone(cwd)
        self.assertIsNone(log)


if __name__ == "__main__":
    unittest.main()
