"""Verify frozen input hashes, numerical generator agreement, and aggregates."""
import argparse
import json
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from threadpoolctl import threadpool_limits
from .benchmarks import FAMILIES
from .models import DATA, case_id, file_hash, load_manifest, load_model, model_hash, plans


# Regeneration may use different libm/BLAS implementations on another host.
# Frozen file and array hashes remain exact; this tolerance applies only to
# comparisons of newly computed float64 values against the frozen arrays.
GENERATOR_RTOL = 64 * np.finfo(np.float64).eps
GENERATOR_ATOL = 64 * np.finfo(np.float64).eps


def assert_generator_matches(generated, frozen, identity):
    """Require identical model structure and float64 agreement to roundoff."""
    if (generated.n_states, generated.n_actions, generated.gamma) != (
            frozen.n_states, frozen.n_actions, frozen.gamma):
        raise AssertionError(identity + ": model dimensions or discount differ")

    def compare_values(actual, expected, label):
        if actual.shape != expected.shape:
            raise AssertionError(identity + ": " + label + " shape differs")
        if not np.all(np.isfinite(actual)) or not np.all(np.isfinite(expected)):
            raise AssertionError(identity + ": " + label + " contains nonfinite values")
        np.testing.assert_allclose(
            actual, expected, rtol=GENERATOR_RTOL, atol=GENERATOR_ATOL,
            equal_nan=False, err_msg=identity + ": " + label + " differs",
        )
        return float(np.max(np.abs(actual - expected))) if actual.size else 0.0

    reward_error = compare_values(generated.rewards, frozen.rewards, "rewards")
    transition_error = 0.0
    for action, (actual, expected) in enumerate(zip(generated.transitions, frozen.transitions)):
        label = "transition action %d" % action
        if sp.issparse(actual) != sp.issparse(expected):
            raise AssertionError(identity + ": " + label + " storage differs")
        if actual.shape != expected.shape:
            raise AssertionError(identity + ": " + label + " shape differs")
        if sp.issparse(actual):
            actual, expected = actual.tocsr(), expected.tocsr()
            np.testing.assert_array_equal(
                actual.indptr, expected.indptr, err_msg=identity + ": CSR row structure differs")
            np.testing.assert_array_equal(
                actual.indices, expected.indices, err_msg=identity + ": CSR columns differ")
            actual, expected = actual.data, expected.data
        # Tiny new transition probabilities must not silently change the model's graph.
        np.testing.assert_array_equal(
            actual != 0, expected != 0, err_msg=identity + ": transition support differs")
        transition_error = max(transition_error, compare_values(actual, expected, label))
    return dict(exact_array_hash_match=model_hash(generated) == model_hash(frozen),
                max_abs_reward_difference=reward_error,
                max_abs_transition_difference=transition_error)


def validate(regenerate_scaling=False, regenerate_public=False):
    manifest = load_manifest()
    for identity, entry in manifest["models"].items():
        path = DATA / entry["file"]
        if file_hash(path) != entry["file_sha256"]:
            raise AssertionError("NPZ hash mismatch: " + identity)
        mdp = load_model(path)
        if model_hash(mdp) != entry["model_sha256"]:
            raise AssertionError("Array hash mismatch: " + identity)
    comparisons = {}
    def compare_generated(generated, spec):
        identity = case_id(spec)
        frozen = load_model(DATA / manifest["models"][identity]["file"])
        comparisons[identity] = assert_generator_matches(generated, frozen, identity)

    with threadpool_limits(limits=1):
        for spec in plans("smoke")[:4]:
            generated = FAMILIES[spec["family"]](n=32, actions=4, gamma=.99, seed=130)
            compare_generated(generated, spec)
        if regenerate_scaling:
            from .scaling_models import random_map, frozen_lake
            for spec in plans("scaling"):
                generated, _ = frozen_lake(random_map(spec["side"], spec["seed"]), spec["gamma"])
                compare_generated(generated, spec)
        if regenerate_public:
            from .public_models import public_model
            names = {"frozenlake4x4": "FrozenLake-4x4", "frozenlake8x8": "FrozenLake-8x8",
                     "taxi": "Taxi-v3", "cliffwalking": "CliffWalking-v0"}
            for spec in plans("public"):
                generated, _ = public_model(names[spec["family"]], spec["gamma"])
                compare_generated(generated, spec)
    archive = json.loads((DATA / "archive_reference.json").read_text())["cases"]
    for identity, row in archive.items():
        if not np.isclose(np.mean(row["queries"]), row["mean_queries"], rtol=0, atol=1e-12):
            raise AssertionError("Archived mean mismatch: " + identity)
        if max(row["queries"]) != row["max_queries"]:
            raise AssertionError("Archived maximum mismatch: " + identity)
    public = [archive[case_id(spec)] for spec in plans("public")]
    if sum(len(row["queries"]) for row in public) != 174 or sum(row["queries"].count(0) for row in public) != 13:
        raise AssertionError("Official archived aggregate counts differ")
    fits = {}
    for gamma in (.99, .999):
        sizes, means = [], []
        for n in (64, 256, 576, 1024, 2304, 4096):
            selected = [archive[case_id(spec)]["mean_queries"] for spec in plans("scaling")
                        if spec["gamma"] == gamma and spec["n"] == n]
            sizes.append(n)
            means.append(float(np.mean(selected)))
        slope, intercept = np.polyfit(np.log(sizes), np.log(means), 1)
        fits[str(gamma)] = dict(slope=float(slope), coefficient=float(np.exp(intercept)))
    return dict(all_passed=True, frozen_models=len(manifest["models"]),
                small_main_generators_matched=4, archived_query_sequences=len(archive),
                official_archived_outer_iterations=174, official_archived_zero_queries=13,
                scaling_regenerated=36 if regenerate_scaling else 0,
                public_regenerated=8 if regenerate_public else 0,
                generator_tolerance=dict(rtol=GENERATOR_RTOL, atol=GENERATOR_ATOL),
                generator_comparisons=comparisons,
                archived_descriptive_fits=fits)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--regenerate-scaling", action="store_true", help="Regenerate all 36 sparse models and compare structure and numerical values; does not solve them.")
    ap.add_argument("--regenerate-public", action="store_true", help="Requires optional Gymnasium 1.1.1; regenerate the eight public models.")
    args = ap.parse_args()
    print(json.dumps(validate(args.regenerate_scaling, args.regenerate_public), indent=2))


if __name__ == "__main__":
    main()
