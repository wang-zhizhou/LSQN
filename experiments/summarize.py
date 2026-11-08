"""Aggregate successful repetitions and descriptive query-count scaling fits."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from .models import DATA, case_id
from .implementations import backend_for


def summarize(directory):
    directory = Path(directory)
    rows = []
    for path in sorted(directory.glob("*_r[0-9]*.json")):
        record = json.loads(path.read_text())
        if "method" in record and "spec" in record:
            rows.append(record)
    protocol = json.loads((directory / "protocol.json").read_text())
    methods = protocol.get("methods", ["LSQN"])
    options = protocol.get("requested_options", {})
    mixed_runs = [dict(case_id=case_id(row["spec"]), method=row["method"], repeat=row["repeat"])
                  for row in rows if row.get("run_id") != protocol.get("run_id")]
    requested_profile = protocol.get("implementation")
    implementation_mismatches = []
    for row in rows:
        if requested_profile not in ("paper", "curves", "reference"):
            continue  # Older preparation outputs predate explicit profiles.
        expected_backend = (backend_for(row["spec"], requested_profile) if row["method"] == "LSQN"
                            else "lsqn_baselines:" + row["method"])
        if row.get("backend") != expected_backend or row.get("implementation") != requested_profile:
            implementation_mismatches.append(dict(case_id=case_id(row["spec"]), method=row["method"],
                repeat=row["repeat"], expected_backend=expected_backend, actual_backend=row.get("backend")))
    archive = json.loads((DATA / "archive_reference.json").read_text())["cases"]
    groups = {}
    for record in rows:
        groups.setdefault((case_id(record["spec"]), record["method"]), []).append(record)
    cases = []
    for (identity, method), repetitions in groups.items():
        repetitions.sort(key=lambda row: row["repeat"])
        first = repetitions[0]
        expected = (1 if protocol.get("suite") == "smoke" else
                    options.get("scaling_repetitions", 1) if first["spec"]["suite"] == "scaling"
                    else options.get("repetitions", 3))
        all_repeats = len(repetitions) == expected and {row["repeat"] for row in repetitions} == set(range(expected))
        same_run = all(row.get("run_id") == protocol.get("run_id") for row in repetitions)
        same_backend = len({row.get("backend") for row in repetitions}) == 1
        valid_backend = not any(row["case_id"] == identity and row["method"] == method
                                for row in implementation_mismatches)
        complete = all_repeats and same_run and same_backend and valid_backend and all(row["success"] for row in repetitions)
        equal_queries = (all(row["queries"] == first["queries"] for row in repetitions)
                         if method == "LSQN" else None)
        equal_inputs = all(row["model"]["model_sha256"] == first["model"]["model_sha256"] for row in repetitions)
        complete = complete and equal_inputs
        entry = dict(case_id=identity, method=method, spec=first["spec"], repetitions=len(repetitions),
                     implementation=first.get("implementation"), backend=first.get("backend"),
                     same_backend_across_repeats=same_backend, matches_requested_backend=valid_backend,
                     expected_repetitions=expected, all_repetitions_present=all_repeats,
                     same_run_id=same_run,
                     all_success=complete, same_query_sequence_across_repeats=equal_queries,
                     same_model_across_repeats=equal_inputs,
                     model_sha256=first["model"]["model_sha256"],
                     median_wall_seconds=float(np.median([row["wall_seconds"] for row in repetitions])) if complete else None,
                     wall_seconds_all=[row["wall_seconds"] for row in repetitions],
                     failed_repeats=[row["repeat"] for row in repetitions if not row["success"]],
                     query_summary=first.get("query_summary"),
                     maximum_final_residual=max((row["independent_verification"]["residual_inf"]
                                                 for row in repetitions if row["independent_verification"]["residual_inf"] is not None), default=None))
        if method == "LSQN" and identity in archive and first.get("backend", "reference") == "reference":
            reference = archive[identity]
            entry["archive_query_comparison"] = dict(
                source_id=reference["source_id"], archived_mean=reference["mean_queries"],
                archived_maximum=reference["max_queries"],
                exact_sequence_match=first["queries"] == reference["queries"],
                note="Cross-platform floating-point trajectories can differ; convergence is checked independently. Archived timing is not compared.")
        cases.append(entry)
    scaling = []
    fits = {}
    for gamma in (.99, .999):
        for n in (64, 256, 576, 1024, 2304, 4096):
            subset = [row for row in cases if row["method"] == "LSQN" and row["spec"]["suite"] == "scaling"
                      and row["spec"]["gamma"] == gamma and row["spec"]["n"] == n]
            if not subset:
                continue
            full = len(subset) == 3 and {row["spec"]["seed"] for row in subset} == {130, 131, 132}
            eligible = full and all(row["all_success"] and row["same_query_sequence_across_repeats"] for row in subset)
            scaling.append(dict(gamma=gamma, n=n, instances=len(subset), complete_seed_group=eligible,
                                mean_instance_queries=float(np.mean([row["query_summary"]["mean"] for row in subset])) if eligible else None))
        eligible_rows = [row for row in scaling if row["gamma"] == gamma and row["complete_seed_group"]]
        if len(eligible_rows) == 6:
            x = np.log([row["n"] for row in eligible_rows])
            y = np.log([row["mean_instance_queries"] for row in eligible_rows])
            slope, intercept = np.polyfit(x, y, 1)
            fits[str(gamma)] = dict(slope=float(slope), coefficient=float(np.exp(intercept)),
                                   caveat="Descriptive six-size query fit, not a runtime exponent or an asymptotic complexity bound.")
    expected_groups = {(case_id(spec), method) for spec in protocol.get("cases", []) for method in methods}
    missing = sorted(expected_groups - set(groups))
    comparisons = []
    for identity in sorted({row["case_id"] for row in cases if row["spec"]["suite"] == "main"}):
        selected = [row for row in cases if row["case_id"] == identity]
        same_inputs = len({row["model_sha256"] for row in selected}) == 1 and all(row["same_model_across_repeats"] for row in selected)
        full_protocol = (protocol.get("suite") == "main" and not mixed_runs and not implementation_mismatches
                         and len(methods) == 10 and len(selected) == 10
                         and all(row["expected_repetitions"] == 3 for row in selected))
        eligible = ([row for row in selected if row["all_success"]]
                    if same_inputs and full_protocol else [])
        eligible.sort(key=lambda row: row["median_wall_seconds"])
        comparisons.append(dict(case_id=identity, same_inputs_across_methods=same_inputs,
                                ranking_requires="all ten methods attempted in the same main-suite batch and model, three successful repetitions per ranked method",
                                rankings=[dict(rank=index + 1, method=row["method"], backend=row["backend"], median_wall_seconds=row["median_wall_seconds"])
                                          for index, row in enumerate(eligible)],
                                failed_or_incomplete_methods=[row["method"] for row in selected if not row["all_success"]],
                                missing_methods=[method for method in methods if (identity, method) not in groups],
                                note="Smoke, selected-method or non-three-repeat runs are validation/exploration and are not assigned formal comparison ranks."))
    return dict(configurations=len({identity for identity, method in groups}), method_configurations=len(cases), executions=len(rows),
                methods=methods, implementation=requested_profile, run_id=protocol.get("run_id"), mixed_run_records=mixed_runs,
                implementation_mismatches=implementation_mismatches,
                missing_case_methods=[dict(case_id=identity, method=method) for identity, method in missing],
                all_success=bool(rows) and not missing and not mixed_runs and not implementation_mismatches and all(row["all_success"] for row in cases),
                counts_use="First repetition per configuration; never triple-count timing repetitions",
                scope="Methods from this one run and requested implementation profile only; optional reference PI diagnostic is excluded from timing comparisons",
                cases=cases, main_comparisons=comparisons, scaling=scaling, descriptive_fits=fits)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("directory", type=Path)
    args = ap.parse_args()
    print(json.dumps(summarize(args.directory), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
