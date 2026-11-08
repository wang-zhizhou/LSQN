"""Run a suite, record the host, and bundle results without overwriting files."""
import argparse
import datetime
import json
import os
from pathlib import Path
import platform
import sys
import traceback
import zipfile

from environment import capture


class Tee:
    def __init__(self, terminal, log):
        self.terminal, self.log = terminal, log

    def write(self, text):
        self.terminal.write(text)
        self.log.write(text)
        return len(text)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=("smoke", "main", "public", "scaling", "all"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--threads", type=int, default=1)
    args, extra = parser.parse_known_args()
    if not (3, 9) <= sys.version_info[:2] < (3, 13):
        parser.error("Use the lsqn-repro Python 3.11 environment, not Python 3.13+.")
    if args.threads < 1:
        parser.error("--threads must be positive.")
    for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
                     "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = str(args.threads)
    os.environ["MKL_DYNAMIC"] = "FALSE"
    os.environ["OMP_DYNAMIC"] = "FALSE"
    root = Path(__file__).resolve().parents[1]
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = (args.output or root / "results" / (platform.system().lower() + "-" + args.suite + "-" + stamp)).resolve()
    archive = out.with_name(out.name + ".zip")
    if out.exists() or archive.exists():
        parser.error("Choose a new output path; existing outputs are never overwritten.")
    metadata = capture(args.threads)
    from experiments.run import main as run
    log_path = out.with_name(out.name + ".console.log")
    out.parent.mkdir(parents=True, exist_ok=True)
    status = 2
    stdout, stderr = sys.stdout, sys.stderr
    with log_path.open("x") as log:
        sys.stdout, sys.stderr = Tee(stdout, log), Tee(stderr, log)
        try:
            status = run(["--suite", args.suite, "--output", str(out), "--threads", str(args.threads)] + extra)
        except KeyboardInterrupt:
            status = 130
            print("Interrupted: completed and partial records will be retained.")
        except SystemExit as exc:
            status = exc.code if isinstance(exc.code, int) else 2
        except Exception:
            traceback.print_exc()
            status = 2
        finally:
            sys.stdout, sys.stderr = stdout, stderr
    out.mkdir(exist_ok=True)
    metadata["exit_status"] = status
    metadata["command_arguments"] = sys.argv[1:]
    with (out / "environment.json").open("x") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    log_path.rename(out / "console.log")
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(out.rglob("*")):
            if path.is_file():
                handle.write(path, path.relative_to(out.parent))
    print("\nRESULT DIRECTORY:", out)
    print("SEND BACK THIS ZIP:", archive)
    print("Exit status:", status, "(0: all requested checks passed; 1: at least one run failed; 2: setup/execution error)")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
