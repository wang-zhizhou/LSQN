"""Run the eight included methods and both LSQN learning suites serially.

AA-I-S-5 and iGMRES-PI have archived results only in this public package.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import zipfile


def run_child(command, cwd):
    """Let a user interrupt finish the child's result-writing cleanup."""
    child = subprocess.Popen(command, cwd=cwd, start_new_session=(os.name == "posix"))
    try:
        return child.wait(), False, False
    except KeyboardInterrupt:
        print("Interrupt received; allowing the current group to save its records.", flush=True)
        if child.poll() is None:
            child.send_signal(signal.SIGINT)
        try:
            return child.wait(timeout=60), True, False
        except (KeyboardInterrupt, subprocess.TimeoutExpired):
            print("Cleanup did not complete; retaining the partial files.", flush=True)
            child.kill()
            return child.wait(), True, True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--implementation", choices=("paper", "curves", "reference"), default="paper")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = (args.output or root / "results" / (platform.system().lower() + "-paper-" + stamp)).resolve()
    archive = out.with_name(out.name + ".zip")
    if out.exists() or archive.exists():
        parser.error("Choose a new output path; existing outputs are never overwritten.")
    groups = [("main", ["--methods", "all"]), ("public", []), ("scaling", [])]
    commands = [(suite, [sys.executable, str(root / "scripts" / "run_suite.py"), suite,
                         "--output", str(out / suite), "--implementation", args.implementation] + extra) for suite, extra in groups]
    if args.dry_run:
        print(json.dumps(dict(total_solves=828, groups=commands), indent=2))
        return 0
    out.mkdir(parents=True)
    statuses = []
    interrupted = False
    try:
        for suite, command in commands:
            print("\nStarting", suite, flush=True)
            status, stopped, forced = run_child(command, cwd=root)
            interrupted = interrupted or stopped
            statuses.append(dict(suite=suite, exit_status=status, forced_termination=forced,
                                 environment_saved=(out / suite / "environment.json").is_file(),
                                 summary_saved=(out / suite / "summary.json").is_file()))
            if stopped or status not in (0, 1):
                print("Stopped after a setup error or interruption; existing records are retained.")
                break
    except KeyboardInterrupt:
        interrupted = True
        print("Interrupted; completed records will be bundled.")
    with (out / "collection.json").open("x") as handle:
        json.dump(dict(schema="lsqn-paper-collection-v1", planned_solves=828,
                       implementation_profile=args.implementation,
                       completed_groups=statuses, interrupted=interrupted,
                       scope="Public subset: 8 methods x 32 cases x 3 runs; public 24 LSQN runs; scaling 36 LSQN runs. AA-I-S-5 and iGMRES-PI are not rerun.",
                       note="Exit 1 preserves numerical failures/timeouts and continues the remaining suites."), handle, indent=2)
        handle.write("\n")
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(out.rglob("*")):
            if path.is_file() and path.suffix != ".zip":
                handle.write(path, path.relative_to(out.parent))
    complete = len(statuses) == len(groups) and not interrupted and all(row["exit_status"] in (0, 1) for row in statuses)
    print("\nSEND BACK THIS " + ("COMPLETE" if complete else "PARTIAL") + " ZIP:", archive)
    if interrupted:
        return 130
    return max([row["exit_status"] for row in statuses] or [2])


if __name__ == "__main__":
    raise SystemExit(main())
