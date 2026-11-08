"""Accept roundoff in regenerated data without concealing changed inputs."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp

from lsqn import MDP
from experiments.models import save_model, file_hash, model_hash
from experiments.validate import assert_generator_matches, validate


def example(sparse=False):
    p = np.array([[0.75, 0.25, 0.0], [0.0, 0.75, 0.25], [0.25, 0.0, 0.75]])
    return MDP([sp.csr_matrix(p) if sparse else p], np.array([[0.1, -0.2, 0.3]]), 0.99)


class InputValidationTests(unittest.TestCase):
    def test_exact_and_roundoff_matches_report_differences(self):
        for sparse in (False, True):
            with self.subTest(sparse=sparse):
                frozen = example(sparse)
                self.assertTrue(assert_generator_matches(frozen, frozen, "case")["exact_array_hash_match"])
                generated = copy.deepcopy(frozen)
                generated.rewards[0, 0] = np.nextafter(generated.rewards[0, 0], np.inf)
                values = generated.transitions[0].data if sparse else generated.transitions[0]
                values.flat[0] = np.nextafter(values.flat[0], np.inf)
                report = assert_generator_matches(generated, frozen, "case")
                self.assertFalse(report["exact_array_hash_match"])
                self.assertGreater(report["max_abs_reward_difference"], 0.0)
                self.assertGreater(report["max_abs_transition_difference"], 0.0)

    def test_material_reward_and_transition_changes_fail(self):
        for sparse in (False, True):
            for field in ("rewards", "transitions"):
                with self.subTest(sparse=sparse, field=field):
                    frozen = example(sparse)
                    generated = copy.deepcopy(frozen)
                    if field == "rewards":
                        generated.rewards[0, 0] += 1e-8
                    else:
                        values = generated.transitions[0].data if sparse else generated.transitions[0][0]
                        values[0] += 1e-8
                        values[1] -= 1e-8
                    with self.assertRaises(AssertionError):
                        assert_generator_matches(generated, frozen, "case")

    def test_structure_changes_fail_even_for_tiny_entries(self):
        frozen = example()
        generated = copy.deepcopy(frozen)
        generated.transitions[0][0, 2] = 1e-16
        with self.assertRaisesRegex(AssertionError, "support"):
            assert_generator_matches(generated, frozen, "case")
        frozen = example(True)
        generated = copy.deepcopy(frozen)
        generated.transitions[0].indices[1] = 2
        with self.assertRaisesRegex(AssertionError, "CSR columns"):
            assert_generator_matches(generated, frozen, "case")
        with self.assertRaisesRegex(AssertionError, "storage"):
            assert_generator_matches(example(True), example(False), "case")

    def test_dimensions_discount_and_nonfinite_values_fail(self):
        frozen = example()
        for value in (np.nan, np.inf):
            generated = copy.deepcopy(frozen)
            generated.rewards[0, 0] = value
            with self.assertRaisesRegex(AssertionError, "nonfinite"):
                assert_generator_matches(generated, frozen, "case")
        generated = copy.deepcopy(frozen)
        generated.gamma = np.nextafter(frozen.gamma, 1.0)
        with self.assertRaisesRegex(AssertionError, "discount"):
            assert_generator_matches(generated, frozen, "case")
        smaller = MDP([np.eye(2)], np.zeros((1, 2)), frozen.gamma)
        with self.assertRaisesRegex(AssertionError, "dimensions"):
            assert_generator_matches(smaller, frozen, "case")

    def test_frozen_archive_hash_stays_strict(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.npz"
            frozen = example()
            save_model(frozen, path)
            manifest = {"models": {"case": {"file": path.name,
                        "file_sha256": file_hash(path), "model_sha256": model_hash(frozen)}}}
            # A tiny edit still counts as corruption of an archived reference.
            frozen.rewards[0, 0] = np.nextafter(frozen.rewards[0, 0], np.inf)
            changed = Path(tmp) / "changed.npz"
            save_model(frozen, changed)
            path.write_bytes(changed.read_bytes())
            with patch("experiments.validate.DATA", Path(tmp)), patch(
                    "experiments.validate.load_manifest", return_value=manifest):
                with self.assertRaisesRegex(AssertionError, "NPZ hash mismatch"):
                    validate()


if __name__ == "__main__":
    unittest.main()
