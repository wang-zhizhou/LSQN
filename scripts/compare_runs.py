"""Compare run inputs, implementations and results; timing ratios are descriptive."""
import argparse
import gzip
import json
from pathlib import Path
import numpy as np


def read(path):
    if not path.is_file():
        path = path.with_suffix(path.suffix + ".gz")
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(path.read_text(encoding="utf-8"))


def records(directory):
    result = {}
    for row in read(directory / "summary.json")["runs"]:
        if "file" in row:
            record = read(directory / row["file"])
            key = (row["case_id"], record.get("method", "LSQN"), row["repeat"])
            result[key] = record
    return result


def compare(left, right):
    a, b = records(left), records(right)
    pa, pb = read(left / "protocol.json"), read(right / "protocol.json")
    rows = []
    for key in sorted(a.keys() & b.keys()):
        x, y = a[key], b[key]
        same_model = x["model"]["model_sha256"] == y["model"]["model_sha256"]
        vx, vy = x["independent_verification"], y["independent_verification"]
        both_converged = bool(x["success"] and y["success"])
        difference = None
        bound = None
        within_bound = None
        if same_model and vx["finite"] and vy["finite"]:
            xx, yy = np.asarray(x["value"]), np.asarray(y["value"])
            difference = float(np.linalg.norm(xx - yy, np.inf))
            bound = (vx["residual_inf"] + vy["residual_inf"]) / (1 - x["spec"]["gamma"])
            roundoff = 100 * np.finfo(float).eps * max(1., np.linalg.norm(xx, np.inf), np.linalg.norm(yy, np.inf))
            within_bound = bool(difference <= bound + roundoff)
        rows.append(dict(case_id=key[0], method=key[1], repeat=key[2],
            implementation=[x.get("implementation"), y.get("implementation")],
            same_implementation=x.get("implementation") == y.get("implementation"),
            backend=[x.get("backend"), y.get("backend")],
            same_backend=x.get("backend") == y.get("backend"),
            same_input_arrays=same_model, both_converged=both_converged,
            value_difference_inf=difference, residual_based_error_bound=bound,
            values_agree_with_error_bound=within_bound,
            same_iterations=x["iterations"] == y["iterations"],
            iterations=[x["iterations"], y["iterations"]],
            same_query_sequence=(x.get("queries") == y.get("queries")) if key[1] == "LSQN" else None,
            policy_vectors_equal=x.get("policy") == y.get("policy"),
            wall_seconds=[x["wall_seconds"], y["wall_seconds"]],
            right_over_left_time=y["wall_seconds"] / x["wall_seconds"] if x["wall_seconds"] else None))
    return dict(schema="lsqn-cross-machine-comparison-v1",
        same_source_hashes=pa["source_sha256"] == pb["source_sha256"],
        same_solver_options=pa["solver_options"] == pb["solver_options"],
        same_implementation_profile=pa.get("requested_options", {}).get("implementation") == pb.get("requested_options", {}).get("implementation"),
        compared_runs=len(rows), unmatched_left=[list(k) for k in sorted(a.keys() - b.keys())],
        unmatched_right=[list(k) for k in sorted(b.keys() - a.keys())],
        same_input_arrays=all(r["same_input_arrays"] for r in rows) if rows else False,
        all_compared_converged=all(r["both_converged"] for r in rows) if rows else False,
        interpretation="Unequal inputs or code need separate explanation. Policy/iteration/query differences alone do not establish incorrectness. Timings are descriptive, not a controlled CPU benchmark.",
        runs=rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--details", action="store_true", help="Print every per-run comparison.")
    args = parser.parse_args()
    output = compare(args.left, args.right)
    if args.output:
        with args.output.open("x") as handle:
            json.dump(output, handle, indent=2, allow_nan=False)
            handle.write("\n")
    shown = output if args.details else {key: value for key, value in output.items() if key != "runs"}
    if not args.details:
        shown["matching_iteration_counts"] = sum(row["same_iterations"] for row in output["runs"])
        shown["value_checks_within_bound"] = sum(row["values_agree_with_error_bound"] is True for row in output["runs"])
        shown["lsqn_matching_query_sequences"] = sum(row["same_query_sequence"] is True for row in output["runs"])
    print(json.dumps(shown, indent=2, allow_nan=False))
