import json
import tempfile
import unittest
from pathlib import Path

from pharm_training.search import build_trials, nested_set, trial_id, write_summary


class SearchWorkflowTests(unittest.TestCase):
    def test_trials_are_deterministic_and_nested_values_apply(self):
        config = {"seed": 7, "parameters": {"learning_rate": [1e-4, 2e-4],
                                              "architecture.depth": [4, 6]}}
        first = build_trials(config)
        self.assertEqual(first, build_trials(config))
        self.assertEqual(len(first), 4)
        target = {}
        nested_set(target, "architecture.depth", 6)
        self.assertEqual(target, {"architecture": {"depth": 6}})
        self.assertEqual(trial_id(first[0]), trial_id(first[0]))

    def test_best_full_config_restores_full_training_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            pilot = {
                "epochs": 120, "early_stopping_patience": 12, "minimum_epochs": 15,
                "train_limit": 50000, "eval_limit": 10000, "evaluate_test": False,
                "learning_rate": 0.0002, "output_dir": "pilot",
            }
            config_path = output / "pilot.json"
            config_path.write_text(json.dumps(pilot))
            base = dict(pilot, epochs=700, early_stopping_patience=20, minimum_epochs=25,
                        train_limit=-1, eval_limit=-1, evaluate_test=True)
            write_summary(output, [{"status": "complete", "best_val_score": "0.8",
                                    "config_path": str(config_path)}], base)
            promoted = json.loads((output / "best_full_config.json").read_text())
            self.assertEqual(promoted["epochs"], 700)
            self.assertEqual(promoted["train_limit"], -1)
            self.assertTrue(promoted["evaluate_test"])
            self.assertEqual(promoted["learning_rate"], 0.0002)


if __name__ == "__main__":
    unittest.main()
