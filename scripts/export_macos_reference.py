"""Export historical Mac evidence without running solvers (standard library only).

The research archive is a separate, local input: it is not downloaded and is not
part of the public package. Exported source IDs are relative to that archive.
"""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics


STUDY = "scb_section6_rerun_2026-10-10"
FINAL = "output/pdf/runtime_final_queue_2026-10-14"
SCALE = "scb_public_benchmarks_2026-09-30/results"


def read(path):
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as stream:
            return json.load(stream)
    return json.loads(path.read_text())


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_method(method):
    return "LSQN" if method == "SCB" else method


def finite(value):
    return value if value is None or math.isfinite(value) else None


class Exporter:
    def __init__(self, archive, package):
        self.archive = archive.resolve()
        self.package = package.resolve()
        self.out = self.package / "benchmarks/macos-m2-paper"
        self.out.mkdir(parents=True, exist_ok=True)
        self.manifest = {}
        self.json_cache = {}

    def relative(self, value):
        text = str(value)
        if str(self.archive) in text:
            return text.replace(str(self.archive) + "/", "")
        # Final provenance has an old absolute checkout path. Resolve by its
        # stable research-tree suffix, without retaining the private prefix.
        for marker in (STUDY + "/", "output/pdf/", "scb_public_benchmarks_2026-09-30/"):
            if marker in text:
                return text[text.index(marker):]
        if text.startswith("/"):
            return "external-library/" + Path(text).name
        return text

    def clean(self, obj):
        if isinstance(obj, dict):
            return {self.relative(k): self.clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self.clean(v) for v in obj]
        if isinstance(obj, str):
            return self.relative(obj)
        if isinstance(obj, float):
            return finite(obj)
        return obj

    def ref(self, source, **extra):
        source = self.relative(source)
        if source not in self.manifest:
            path = self.archive / source
            self.manifest[source] = {"source_id": source, "sha256": digest(path),
                                     "original_size_bytes": path.stat().st_size}
        return dict(self.manifest[source], **extra)

    def load(self, source):
        source = self.relative(source)
        if source.endswith(".json.gz"):
            return read(self.archive / source)
        if source not in self.json_cache:
            self.json_cache[source] = read(self.archive / source)
        return self.json_cache[source]

    def write(self, name, obj):
        path = self.out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.clean(obj), indent=2, allow_nan=False) + "\n")

    def write_lines(self, name, records):
        with (self.out / name).open("w") as stream:
            for record in records:
                stream.write(json.dumps(self.clean(record), allow_nan=False) + "\n")

    def snapshot(self, source, name, obj=None):
        ref = self.ref(source)
        self.write("sources/" + name, self.load(source) if obj is None else obj)
        self.manifest[ref["source_id"]]["bundled_normalized_copy"] = "sources/" + name

    def run_summary(self, raw, source, suite, family, n, gamma, method, repeat, variant=None):
        residual = raw.get("independent_residual_inf", raw.get("independent_residual"))
        verified = residual is not None and math.isfinite(residual) and residual <= 1e-8
        checks = raw.get("validation", {})
        success = bool(raw.get("converged") and verified and checks.get("all", True))
        queries = None
        if method == "LSQN":
            queries = raw.get("queries")
            if queries is None and "inner_calls" in raw:
                queries = [c["queries"] for c in raw["inner_calls"]]
            if queries is None and "history" in raw:
                queries = [h["enrichments"] for h in raw["history"]]
        total = raw.get("stats", {}).get("enrichments") if method == "LSQN" else None
        if queries is not None:
            assert total is None or total == sum(queries), (source, total, sum(queries))
            total = sum(queries)
        return dict(suite=suite, family=family, n=n, gamma=gamma, method=method,
                    original_method=raw.get("method", "SCB" if method == "LSQN" else method),
                    repeat=repeat, seed=raw.get("seed", raw.get("spec", {}).get("seed")),
                    model_sha256=raw.get("model_sha256", raw.get("model", {}).get("model_sha256")),
                    original_model_hash=raw.get("model_hash"),
                    variant=variant, converged=raw.get("converged"), success=success,
                    message=raw.get("message"), timeout=raw.get("timeout"),
                    wall_seconds=raw.get("wall_s", raw.get("wall_seconds")),
                    wall_budget_seconds=raw.get("wall_budget_s"),
                    outer_iterations=raw.get("iterations"), total_queries=total,
                    queries=queries, residual_inf=finite(residual),
                    residual_nonfinite=residual is not None and not math.isfinite(residual),
                    original_validation=checks or None,
                    source_ref=self.ref(source))

    @staticmethod
    def aggregate(runs):
        times = [r["wall_seconds"] for r in runs]
        outer = [r["outer_iterations"] for r in runs]
        queries = [r["total_queries"] for r in runs]
        success = all(r["success"] for r in runs)
        return dict(all_success=success, successes=sum(r["success"] for r in runs),
                    repetitions=len(runs), wall_seconds_all=times,
                    median_wall_seconds=statistics.median(times) if success else None,
                    raw_elapsed_median_seconds=statistics.median(times),
                    outer_iterations=statistics.median(outer) if None not in outer else None,
                    outer_iterations_all=outer,
                    total_queries=statistics.median(queries) if None not in queries else None,
                    total_queries_all=queries, messages=[r["message"] for r in runs],
                    success_all=[r["success"] for r in runs],
                    residual_inf_all=[r["residual_inf"] for r in runs])

    def main_table(self):
        cells = self.load(FINAL + "/cell_provenance.json")
        rows, runs, curves = [], [], []
        for cell_index, cell in enumerate(cells):
            family, n, gamma = cell["family"], cell["n"], cell["gamma"]
            method = canonical_method(cell["method"])
            source = self.relative(cell["source"])
            directory = str(Path(source).parent)
            if "section6_optimized_2026-10-11" in source:
                summary = next(x for x in self.load(source) if
                    (x["family"], x["n"], x["gamma"], x["method"]) ==
                    (family, n, gamma, cell["method"]))
                paths = [STUDY + "/" + p for p in summary["source_files"]]
                # The legacy summary may list repetitions in arbitrary order.
                paths.sort(key=lambda p: int(p.rsplit("_r", 1)[1].split("_", 1)[0].split(".", 1)[0]))
            elif method == "LSQN":
                variant = {"cluster": "SCB", "latent": "native",
                           "garnet": "fused_sparse", "queue": "native_query"}[family]
                paths = [f"{directory}/{family}_n{n}_g{gamma}_{variant}_r{rep}.json.gz"
                         for rep in range(3)]
            else:
                paths = [f"{directory}/{family}_n{n}_g{gamma}_{cell['method']}_r{rep}.json.gz"
                         for rep in range(3)]
            selected = []
            traces = {}
            for path in paths:
                raw = self.load(path)
                run = self.run_summary(raw, path, "main", family, n, gamma,
                                       method, raw["rep"], cell.get("variant"))
                selected.append(run)
                traces[raw["rep"]] = ([dict(k=point["k"], residual2=point["residual2"])
                                      for point in raw["trace"]] if raw.get("trace") is not None else None)
            assert [r["repeat"] for r in selected] == [0, 1, 2]
            aggregate = self.aggregate(selected)
            assert aggregate["all_success"] == cell["success"], (family, n, gamma, method)
            assert aggregate["median_wall_seconds"] == cell["seconds"], (family, n, gamma, method)
            rows.append(dict(family=family, n=n, gamma=gamma, seed=130, method=method,
                             original_method=cell["method"], variant=cell.get("variant"),
                             model_sha256=selected[0]["model_sha256"],
                             model_sha256_all=[r["model_sha256"] for r in selected],
                             queries=selected[0]["queries"],
                             queries_all=[r["queries"] for r in selected],
                             environment_source_id=("output/pdf/section6_optimized_2026-10-11/final_protocol.json"
                                if "section6_optimized_2026-10-11" in source else directory + "/protocol.json"),
                             **aggregate, rank=cell["display_table_rank"],
                             source_refs=[self.ref(source)] + [r["source_ref"] for r in selected]))
            runs.extend(selected)
            chosen = sorted(selected, key=lambda r:(r["wall_seconds"], r["repeat"]))[1]
            trace = traces[chosen["repeat"]]
            curves.append(dict(family=family, n=n, gamma=gamma, method=method,
                selected_repeat=chosen["repeat"], all_success=aggregate["all_success"],
                wall_seconds=chosen["wall_seconds"], trace=trace, source_ref=chosen["source_ref"],
                selection="Median observed elapsed time, ordered by (wall_seconds, repeat), including failed triples.",
                initial_point_recorded=bool(trace and trace[0]["k"] == 0),
                terminal_point_recorded=bool(trace and trace[-1]["k"] == chosen["outer_iterations"]),
                outer_iterations=chosen["outer_iterations"], selected_run_success=chosen["success"],
                message=chosen["message"]))
            if (cell_index + 1) % 40 == 0:
                print("Read and checked", cell_index + 1, "of 320 historical cells.", flush=True)
        assert len(rows) == 320 and len(runs) == 960
        groups = defaultdict(list)
        for row in rows:
            groups[row["family"], row["n"], row["gamma"]].append(row)
        for key, group in groups.items():
            assert len(group) == 10
            successful = sorted([r for r in group if r["all_success"]], key=lambda r:r["median_wall_seconds"])
            for row in successful:
                positions = [i + 1 for i, r in enumerate(successful)
                             if r["median_wall_seconds"] == row["median_wall_seconds"]]
                assert row["rank"] == statistics.mean(positions), (key, row["method"])
            failure_rank = (len(successful) + 1 + len(group)) / 2
            assert all(r["rank"] == failure_rank for r in group if not r["all_success"])
        leaderboard = []
        for method in sorted({r["method"] for r in rows}):
            selected = [r for r in rows if r["method"] == method]
            leaderboard.append(dict(method=method, mean_rank=statistics.mean(r["rank"] for r in selected),
                successes=sum(r["all_success"] for r in selected), wins=sum(r["rank"] == 1 for r in selected)))
        leaderboard.sort(key=lambda r:r["mean_rank"])
        original = self.load(FINAL + "/ranking_audit.json")
        for row in leaderboard:
            expected = next(r for r in original["leaderboard"] if canonical_method(r["method"]) == row["method"])
            assert all(row[k] == expected[k] for k in ("mean_rank", "successes", "wins"))
        assert next(r["mean_rank"] for r in leaderboard if r["method"] == "LSQN") == 3.53125
        self.write("main.json", dict(schema="lsqn-macos-paper-main-v1", records=rows,
            leaderboard=leaderboard, configurations=32, executions=960,
            note="Historical mixed-batch paper table; not a fresh simultaneous ten-method benchmark.",
            ranking_rule="Successful methods ranked by unrounded median runtime; failed triples tie below all successes at the mean remaining position.",
            failure_time_rule="Failed median_wall_seconds is null; all observed elapsed times and messages remain available.",
            source_refs=[self.ref(FINAL + "/cell_provenance.json"), self.ref(FINAL + "/ranking_audit.json")]))
        self.write_lines("main_runs.jsonl", runs)
        with gzip.GzipFile(filename=str(self.out / "main_curves.jsonl.gz"), mode="wb", mtime=0) as stream:
            for curve in curves:
                stream.write((json.dumps(self.clean(curve), allow_nan=False) + "\n").encode())
        print("main.json and main_curves.jsonl.gz are complete.", flush=True)
        self.snapshot(FINAL + "/cell_provenance.json", "final_cell_provenance.json")
        self.snapshot(FINAL + "/ranking_audit.json", "final_ranking_audit.json")
        for index, source in enumerate(sorted({self.relative(c["source"]) for c in cells})):
            self.snapshot(source, f"batch_summary_{index}.json")
        return rows

    def learning(self):
        refpath = self.package / "experiments/data/archive_reference.json"
        references = read(refpath)["cases"]
        public_source = STUDY + "/supplement/public_official.json.gz"
        public_raw = self.load(public_source)
        records = {"public": [], "scaling": []}
        all_runs = []
        for case_id, reference in references.items():
            suite = case_id.split("_", 1)[0]
            source = reference["source_id"]
            assert self.ref(source)["sha256"] == reference["source_sha256"]
            if suite == "public":
                family = case_id.split("_")[1]
                n = int(case_id.split("_n")[1].split("_")[0])
                gamma = float(case_id.split("_g")[1].split("_")[0].replace("p", "."))
                matched = sorted([r for r in public_raw if
                    r["public_model"]["augmented_states"] == n and
                    r["public_model"]["discount"] == gamma], key=lambda r:r["rep"])
                assert len(matched) == 3
            else:
                matched = [self.load(source)]
                family = "frozenlake"
                n = matched[0]["model"]["n_states"]
                gamma = matched[0]["spec"]["gamma"]
            runs = [self.run_summary(r, source, suite, family, n, gamma, "LSQN", r["rep"], "reference")
                    for r in matched]
            assert all(r["queries"] == reference["queries"] for r in runs)
            first = runs[0]
            aggregate = self.aggregate(runs)
            row = dict(case_id=case_id, suite=suite, family=family, n=n, gamma=gamma,
                seed=0 if suite == "public" else matched[0]["spec"]["seed"], method="LSQN", backend="reference",
                **aggregate, queries=first["queries"], mean_queries=reference["mean_queries"],
                max_queries=reference["max_queries"], zero_queries=first["queries"].count(0),
                full_dimension_queries=first["queries"].count(n), source_refs=[self.ref(source)],
                archive_query_reference="experiments/data/archive_reference.json#cases/" + case_id)
            records[suite].append(row)
            for run in runs:
                run["case_id"] = case_id
            all_runs.extend(runs)
        assert len(records["public"]) == 8 and len(records["scaling"]) == 36 and len(all_runs) == 60
        for suite in records:
            self.write(suite + ".json", dict(schema="lsqn-macos-paper-learning-v1", records=records[suite],
                query_reference=dict(package_path="experiments/data/archive_reference.json", sha256=digest(refpath)),
                note="Queries copied from historical raw records and checked exactly against the bundled query archive."))
        self.write_lines("learning_runs.jsonl", all_runs)
        self.snapshot(STUDY + "/supplement/public_summary.json", "official_public_summary.json")
        self.snapshot(SCALE + "/summary.json", "original_scaling_analysis.json",
            {k:v for k,v in self.load(SCALE + "/summary.json").items() if k in ("scaling", "descriptive_fits", "caveat")})

    def environments(self, main_rows):
        sources = {
            "baseline_paper_table": "output/pdf/section6_optimized_2026-10-11/final_protocol.json",
            "original_main_batch": STUDY + "/protocol.json",
            "scaling": SCALE + "/scale/protocol.json",
        }
        for row in main_rows:
            if row["method"] == "LSQN":
                sources[row["family"] + "_final_runtime"] = str(Path(row["source_refs"][0]["source_id"]).parent / "protocol.json")
        entries = []
        for batch, source in sources.items():
            raw = self.load(source)
            fields = {k:raw.get(k) for k in ("date", "platform", "processor", "python", "numpy", "scipy", "gymnasium")}
            fields.update(cpu_model=None, memory_bytes=None, c_compiler=None, c_compiler_flags=None,
                          threadpoolctl_version=None, threadpools=raw.get("threadpools"))
            snapshot = {k:v for k,v in raw.items() if k not in ("sources", "snapshots", "source_snapshots")}
            self.snapshot(source, "protocol_" + batch + ".json", snapshot)
            entries.append(dict(batch=batch, fields=fields, source_ref=self.ref(source),
                missing_policy="Null means not recorded in this batch protocol; values are not borrowed from another batch or the current host."))
        entries.append(dict(batch="official_public", fields=dict(platform=None, processor=None,
            cpu_model=None, memory_bytes=None, python=None, numpy=None, scipy=None,
            gymnasium=sorted({r["public_model"]["gymnasium"] for r in self.load(STUDY + "/supplement/public_official.json.gz")}),
            threadpoolctl_version=None, threadpools=None, c_compiler=None, c_compiler_flags=None),
            source_ref=self.ref(STUDY + "/supplement/public_official.json.gz"),
            missing_policy="Only Gymnasium's version is embedded in these official-model raw records; missing environment fields remain null."))
        self.write("environment.json", dict(schema="lsqn-historical-environment-v1", records=entries,
            label_note="macos-m2-paper is the supplied archive label. The inspected protocols record arm/macOS; an exact CPU model is not recorded and remains null.",
            current_machine_inspected=False))

    def finish(self):
        self.write("source_manifest.json", dict(schema="lsqn-historical-sources-v1",
            sources=sorted(self.manifest.values(), key=lambda r:r["source_id"]),
            note="SHA-256 values refer to original source bytes, including gzip containers. Normalized copies remove absolute installation paths and omit source-code snapshots from protocol extracts. Per-run summary JSONL files preserve selected historical measurements and omit large solver vectors and histories; main_curves.jsonl.gz separately retains complete selected residual traces."))
        readme = """# Historical Mac paper reference

This directory exports the Mac results used in the paper. It does not rerun any
solver. `main.json` contains 32 configurations × 10 methods; `main_runs.jsonl`
contains the 960 original run measurements selected for those cells. The old
method label SCB is exposed as LSQN, with the original label retained.

`main_curves.jsonl.gz` retains the complete recorded residual-2 trace for the
middle repetition ordered by (whole-solver elapsed time, repeat index) in each
of those same 320 final-runtime cells. Failed triples are retained and labelled.
No curve from the separate block32 paper-figure profile is substituted. Initial
and terminal points are retained as recorded; explicit flags identify whether
the original trace includes k=0 and the recorded final iteration. No missing
trace point is reconstructed or invented.

The main table combines historical batches. Of the 288 baseline cells, 192 come
from the October 11 table (including the authors' iGMRES supplemental batch)
and 96 come from the October 14 targeted leader recheck. The final LSQN choices
are Queue `native_query`, Cluster `cache_reuse`, Garnet `fused_sparse`, and Latent
`native`, from their October 14 confirmation batches. No fastest repetition or
per-case variant is selected. This is a descriptive mixed-batch ranking, not a
fresh simultaneous comparison. Recomputing all 320 ranks gives LSQN 3.53125.

Every cell preserves all three timings, successes, failure messages, iterations,
and available LSQN query counts. Failed cells have null `median_wall_seconds`;
their observed elapsed times remain in `wall_seconds_all`. Missing timeout flags
remain null; the original failure message is retained. `outer_iterations` and
`total_queries` are medians, with all individual values in the corresponding
`*_all` fields. Query counts for other methods are null because they are not the
LSQN protected-learning query metric.

`public.json` contains the eight official Gymnasium configurations, each with
three runs. `scaling.json` contains 36 generated FrozenLake instances (six sizes,
two discounts, three seeds, one run each). `learning_runs.jsonl` retains their
query sequences and measurements. Query sequences are checked against the
existing `experiments/data/archive_reference.json`. Scaling is synthetic, and
its fitted growth exponents describe these six sizes only; they are not
asymptotic complexity bounds. Time means whole-solver elapsed time, not only
inner learning time.

`source_manifest.json` maps relative research-archive IDs to original SHA-256
hashes. `sources/` includes normalized table/protocol evidence. The per-run
summary files omit large value vectors and histories; the selected complete
residual traces are stored separately in `main_curves.jsonl.gz`. Original gzip
hashes can be checked against the separate research archive. No private home
directory is embedded. Historical system library paths are reduced to filenames.

`environment.json` uses only records made during the historical batches. Missing
fields are null; no current-machine information is substituted. In particular,
the exact CPU model and native compiler/flags are absent from the inspected
protocols. The directory name is an archive label, not a new hardware probe.

Maintainers with the separate original research archive can regenerate this
export using the standard library only:

    python scripts/export_macos_reference.py --archive-root /path/to/research-archive

This command reads the archive and writes only this benchmark directory. It
requires the original archive; a public checkout alone does not contain all
original raw files. The checked-in JSON files are directly usable without it.
"""
        (self.out / "README.md").write_text(readme)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", required=True, type=Path)
    parser.add_argument("--package-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    exporter = Exporter(args.archive_root, args.package_root)
    rows = exporter.main_table()
    exporter.learning()
    exporter.environments(rows)
    exporter.finish()
    print("Exported 320 main cells / 960 runs, 8 public cells / 24 runs, and 36 scaling runs.")
    print("Verified every main median and rank; historical LSQN mean rank is 3.53125.")


if __name__ == "__main__":
    main()
