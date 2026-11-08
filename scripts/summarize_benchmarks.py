"""Rebuild the archived macOS/Linux comparison without rerunning any solver.

Only Python's standard library is required. Use --check to verify the committed
comparison JSON and Markdown against the archived inputs.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks"
LINUX = BENCH / "linux-intel-2026-10-25"
MAC = BENCH / "macos-m2-paper"
METHODS = ["LSQN", "VI", "PI", "MPI-4", "MPI-20", "QPI-U", "QPI-BT", "QPI-S", "iGMRES-PI", "AA-I-S-5"]
FAMILIES = ["queue", "cluster", "garnet", "latent"]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def equivalent(a, b):
    """Allow libm rounding in recomputed floats; source hashes remain exact."""
    if isinstance(a, float) and isinstance(b, float):
        return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-14)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(equivalent(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(equivalent(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def key(row):
    return row["family"], row["n"], row["gamma"]


def ranks(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[key(row)].append(row)
    by_method = defaultdict(list)
    by_family = defaultdict(lambda: defaultdict(list))
    success = defaultdict(int)
    wins = defaultdict(int)
    case_ranks = []
    for case, entries in sorted(groups.items()):
        if len(entries) != 10 or {r["method"] for r in entries} != set(METHODS):
            raise ValueError("Missing or duplicate methods: %r" % (case,))
        good = sorted((r for r in entries if r["all_success"]), key=lambda r: r["median_wall_seconds"])
        result = {}
        i = 0
        while i < len(good):
            j = i + 1
            while j < len(good) and good[j]["median_wall_seconds"] == good[i]["median_wall_seconds"]:
                j += 1
            for r in good[i:j]:
                result[r["method"]] = ((i + 1) + j) / 2
                success[r["method"]] += 1
                wins[r["method"]] += int(i == 0)
            i = j
        failure_rank = (len(good) + 1 + len(entries)) / 2
        for row in entries:
            rank = result.get(row["method"], failure_rank)
            by_method[row["method"]].append(rank)
            by_family[case[0]][row["method"]].append(rank)
            result[row["method"]] = rank
        case_ranks.append(dict(family=case[0], n=case[1], gamma=case[2], ranks=result))
    if len(groups) != 32:
        raise ValueError("Expected 32 main benchmark configurations")
    overall = sorted([dict(method=m, mean_rank=statistics.mean(by_method[m]),
                           successful_configurations=success[m], fastest_configurations=wins[m])
                      for m in METHODS], key=lambda r: (r["mean_rank"], r["method"]))
    family = {f: {m: statistics.mean(values) for m, values in ms.items()} for f, ms in by_family.items()}
    return dict(overall=overall, by_family=family, cases=case_ranks)


def fit(rows, field):
    x = [math.log(r["n"]) for r in rows]
    y = [math.log(r[field]) for r in rows]
    xm, ym = statistics.mean(x), statistics.mean(y)
    slope = sum((a - xm) * (b - ym) for a, b in zip(x, y)) / sum((a - xm) ** 2 for a in x)
    intercept = ym - slope * xm
    return dict(slope=slope, coefficient=math.exp(intercept))


def build():
    analyses = {s: read(LINUX / s / "analysis.json") for s in ("main", "public", "scaling")}
    summaries = {s: read(LINUX / s / "summary.json") for s in analyses}
    mac = read(MAC / "main.json")["records"]
    linux = [dict(family=r["spec"]["family"], n=r["spec"]["n"], gamma=r["spec"]["gamma"],
                  method=r["method"], all_success=r["all_success"],
                  median_wall_seconds=r["median_wall_seconds"], query_summary=r.get("query_summary"))
             for r in analyses["main"]["cases"]]
    for suite, expected in [("main", 960), ("public", 24), ("scaling", 36)]:
        if len(summaries[suite]["runs"]) != expected:
            raise ValueError("Unexpected run count for " + suite)
    archive = read(ROOT / "experiments/data/archive_reference.json")["cases"]
    totals = {}
    for suite, summary in summaries.items():
        runs = summary["runs"]
        totals[suite] = dict(runs=len(runs), successful=sum(bool(r["success"]) for r in runs),
                             timeouts=sum(bool(r["timeout"]) for r in runs),
                             other_unsuccessful=sum(not r["success"] and not r["timeout"] for r in runs),
                             lsqn_runs=sum(r["method"] == "LSQN" for r in runs),
                             lsqn_successful=sum(r["method"] == "LSQN" and r["success"] for r in runs))
    mac_ls = {key(r): r for r in mac if r["method"] == "LSQN"}
    main_pairs = []
    for row in linux:
        if row["method"] != "LSQN":
            continue
        previous = mac_ls[key(row)]
        q = row["query_summary"]
        main_pairs.append(dict(family=row["family"], n=row["n"], gamma=row["gamma"],
             macos_seconds=previous["median_wall_seconds"], linux_seconds=row["median_wall_seconds"],
             linux_over_macos=row["median_wall_seconds"] / previous["median_wall_seconds"],
             macos_outer_iterations=previous.get("outer_iterations"), linux_outer_iterations=q["completed_outer_iterations"],
             macos_total_queries=previous.get("total_queries"), linux_total_queries=q["total"],
             same_counts=(previous.get("outer_iterations") == q["completed_outer_iterations"]
                          and previous.get("total_queries") == q["total"])))
    main_pairs.sort(key=lambda r: (FAMILIES.index(r["family"]), r["n"], r["gamma"]))
    public, scaling_cases = [], []
    for suite, target in [("public", public), ("scaling", scaling_cases)]:
        for r in analyses[suite]["cases"]:
            old = archive[r["case_id"]]
            q = r["query_summary"]
            target.append(dict(case_id=r["case_id"], family=r["spec"]["family"], n=r["spec"]["n"],
                  gamma=r["spec"]["gamma"], seed=r["spec"]["seed"], all_success=r["all_success"],
                  macos_mean_queries=statistics.mean(old["queries"]), linux_mean_queries=q["mean"],
                  macos_max_queries=max(old["queries"]), linux_max_queries=q["maximum"],
                  macos_outer_iterations=len(old["queries"]), linux_outer_iterations=q["completed_outer_iterations"],
                  exact_query_sequence_match=r["archive_query_comparison"]["exact_sequence_match"]))
    public.sort(key=lambda r: (r["n"], r["family"], r["gamma"]))
    grouped = defaultdict(list)
    for r in scaling_cases:
        grouped[(r["gamma"], r["n"])].append(r)
    scaling = []
    for (gamma, n), rows in sorted(grouped.items()):
        if len(rows) != 3 or {r["seed"] for r in rows} != {130, 131, 132}:
            raise ValueError("Incomplete scaling seed group")
        scaling.append(dict(gamma=gamma, n=n,
            macos_mean_queries=statistics.mean(r["macos_mean_queries"] for r in rows),
            linux_mean_queries=statistics.mean(r["linux_mean_queries"] for r in rows)))
    fits = {str(g): {platform: fit([r for r in scaling if r["gamma"] == g], platform + "_mean_queries")
                     for platform in ("macos", "linux")} for g in (0.99, 0.999)}
    paths = [MAC / "main.json", ROOT / "experiments/data/archive_reference.json"]
    paths += [LINUX / s / name for s in analyses for name in ("analysis.json", "summary.json")]
    return dict(schema="lsqn-cross-platform-archive-comparison-v1", totals=totals,
                macos_main=ranks(mac), linux_main=ranks(linux), main_ls = main_pairs,
                matching_main_ls_counts=sum(r["same_counts"] for r in main_pairs),
                public=public, scaling_cases=scaling_cases, scaling=scaling, query_growth_fits=fits,
                source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                interpretation="Independent software-platform replication, not an isolated CPU benchmark or a bitwise reproduction. Failed methods rank jointly below successful methods; all timing repetitions must succeed. Scaling fits are descriptive query fits, not complexity bounds.")


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"] +
                     ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def render(x):
    mac = {r["method"]: r for r in x["macos_main"]["overall"]}
    linux = x["linux_main"]["overall"]
    rank_table = table(["Method", "macOS mean rank", "Linux mean rank", "Linux successful cases"],
        [(r["method"], f'{mac[r["method"]]["mean_rank"]:.3f}', f'{r["mean_rank"]:.3f}',
          f'{r["successful_configurations"]}/32') for r in linux])
    family_table = table(["Family", "macOS LSQN mean rank", "Linux LSQN mean rank"],
        [(f.title(), f'{x["macos_main"]["by_family"][f]["LSQN"]:.3f}',
          f'{x["linux_main"]["by_family"][f]["LSQN"]:.3f}') for f in FAMILIES])
    public_table = table(["Environment", "n", "Discount", "Mean queries: macOS / Linux", "Max queries: macOS / Linux"],
        [(r["family"], r["n"], r["gamma"], f'{r["macos_mean_queries"]:.2f} / {r["linux_mean_queries"]:.2f}',
          f'{r["macos_max_queries"]} / {r["linux_max_queries"]}') for r in x["public"]])
    fit_table = table(["Discount", "macOS query-growth exponent", "Linux query-growth exponent"],
        [(g, f'{v["macos"]["slope"]:.4f}', f'{v["linux"]["slope"]:.4f}') for g,v in x["query_growth_fits"].items()])
    timing_table = table(["Family", "n", "Discount", "macOS seconds", "Linux seconds", "Outer iterations: macOS / Linux", "Total queries: macOS / Linux"],
        [(r["family"], r["n"], r["gamma"], f'{r["macos_seconds"]:.6g}', f'{r["linux_seconds"]:.6g}',
          f'{r["macos_outer_iterations"]} / {r["linux_outer_iterations"]}',
          f'{r["macos_total_queries"]} / {r["linux_total_queries"]}') for r in x["main_ls"]])
    return f'''# Cross-platform results

These archives compare the paper's Apple M2/macOS measurements with a separate
Intel Core i9-14900K/Linux run of the release candidate on 25 October 2026.
They support software portability and broadly consistent experimental findings;
they are not a controlled comparison of CPU performance.

## Results at a glance

- Linux completed all 1,020 planned executions: 960 main, 24 public, and 36 scaling.
- LSQN succeeded in all 156 executions. Among the baselines, 51 runs timed out
  and three AA-I-S-5 runs reached the 30,000-iteration limit; these remain in the archive.
- LSQN has the second-lowest mean runtime rank on both platforms. Its advantage
  is strongest on Cluster; PI is fastest on Queue, and QPI-U/BT are faster on
  Garnet and Latent. An overall rank does not imply superiority on most cases.
- LSQN outer-iteration and total-query counts match the historical timing records
  in {x["matching_main_ls_counts"]}/32 main configurations. Exact trajectories need not match across platforms.

## Runtime ranks

For each configuration, take the median of three runs only if all three succeed.
Rank successful methods by this median; failed methods share the mean of the
remaining ranks. Average these ranks equally over the 32 configurations.
Lower is better. This is a mean rank, not a mean wall time or a significance test.

{rank_table}

{family_table}

## Inner learning

Public results compare per-outer-iteration query counts, including zero-query steps.

{public_table}

For scaling, first average queries over the iterations of each instance, then
average the three instance means equally at each size. Fit log(mean queries)
against log(n) over the six sizes. These empirical exponents are not asymptotic
complexity bounds and do not establish that every run entered a superlinear regime.

{fit_table}

## Environment and comparison limits

- Both runs use CPUs; the Linux machine's GPU was not used. The Linux logs record
  one thread in each BLAS library and no fixed CPU affinity.
- Linux used Python 3.11.17, NumPy 1.26.4, and SciPy 1.13.0. Its NumPy OpenBLAS
  0.3.23.dev selected **Prescott**, while SciPy OpenBLAS 0.3.26.dev selected
  **Haswell**. This is the configuration actually measured, not a retuned run.
  `OPENBLAS_CORETYPE` was not recorded. The difference warrants checking before
  interpreting these timings as fully tuned hardware performance.
- macOS values are the paper's final table assembled from several recorded
  batches, rather than one fresh run of this package. Per-cell source identifiers
  and hashes are retained. Missing historical environment fields stay unknown.
- Comparing the Linux main-model hashes with the Mac reference-stage generator
  records gave matching Queue/Garnet hashes and differing Cluster/Latent hashes;
  this does not verify the historical paper timing inputs byte for byte. Linux
  did not retain those full main arrays, so the size or cause of the differences is unresolved.
  Public/scaling inputs are frozen in the package; some trajectories differ even
  for identical inputs. This comparison is not a bitwise reproduction claim.
- Repeated runs measure timing variation on fixed instances; they are not
  independent random model samples. Do not mix selected timings from the two hosts
  into a single performance table or attribute a speed difference solely to CPU.

## Inspect and reproduce the comparison

- [Linux archive](linux-intel-2026-10-25/): all 1,020 run records, including failures,
  plus the recorded protocols, summaries, and environment metadata. Run JSON files
  are gzip-compressed; machine-specific paths are redacted. The import manifest
  records original and released hashes and the redaction policy.
- [macOS paper reference](macos-m2-paper/): per-configuration and per-run source
  extracts, historical environment records, and provenance. It is a reference
  export, not a new macOS execution.
- [Machine-readable comparison](comparison.json) and [audit scope](audit.json).
- [Suggested Section 6 text](../docs/section6_cross_platform.tex).

From the repository root, with Python 3.9 or newer:

```bash
python scripts/summarize_benchmarks.py --check
python scripts/verify_manifest.py
```

To regenerate this page and `comparison.json`, run
`python scripts/summarize_benchmarks.py`. This reads archived measurements only;
it does not run an experiment. See the [main README](../README.md) to run new experiments.

## LSQN per-configuration comparison

Times below are seconds (median of three runs). The machine-readable files retain
the measurements for all ten methods. These host-to-host times are descriptive.

{timing_table}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = build()
    outputs = {BENCH / "comparison.json": json.dumps(result, indent=2, allow_nan=False) + "\n",
               BENCH / "README.md": render(result)}
    for path, content in outputs.items():
        if args.check:
            matches = path.is_file() and (equivalent(read(path), result)
                if path.suffix == ".json" else path.read_text(encoding="utf-8") == content)
            if not matches:
                raise SystemExit("Missing or stale benchmark summary: " + str(path.relative_to(ROOT)))
        else:
            path.write_text(content, encoding="utf-8")
    print("Verified archived cross-platform summaries." if args.check else "Rebuilt benchmarks/README.md and comparison.json.")


if __name__ == "__main__":
    main()
