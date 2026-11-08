"""Check the public package's executable scope and results-only methods."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from experiments.run import main as run
from lsqn_baselines import METHODS, RESULTS_ONLY_METHODS, solve


class PublicScopeTests(unittest.TestCase):
    def test_results_only_methods_fail_before_execution(self):
        self.assertEqual(METHODS, ("VI", "PI", "MPI-4", "MPI-20", "QPI-U", "QPI-BT", "QPI-S"))
        for method in RESULTS_ONLY_METHODS:
            with self.subTest(method=method):
                with self.assertRaisesRegex(ValueError, "archived results only"):
                    solve(method, None)
                with tempfile.TemporaryDirectory() as temp:
                    output = Path(temp) / "not-created"
                    message = StringIO()
                    with redirect_stderr(message), self.assertRaises(SystemExit) as error:
                        run(["--suite", "main", "--methods", method.lower(), "--output", str(output)])
                    self.assertEqual(error.exception.code, 2)
                    self.assertIn("archived results only", message.getvalue())
                    self.assertFalse(output.exists())

    def test_all_records_only_included_methods(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "prepared"
            with redirect_stdout(StringIO()):
                status = run(["--suite", "smoke", "--methods", "all", "--prepare-only", "--output", str(output)])
            self.assertEqual(status, 0)
            protocol = json.loads((output / "protocol.json").read_text())
            self.assertEqual(protocol["methods"], ["LSQN", *METHODS])
            self.assertEqual(len(protocol["cases"]), 6)
            self.assertTrue(set(RESULTS_ONLY_METHODS).isdisjoint(protocol["methods"]))

    def test_paper_dry_run_counts_public_subset(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run([sys.executable, str(root / "scripts/run_paper.py"), "--dry-run"],
                                cwd=root, capture_output=True, text=True, check=True)
        plan = json.loads(result.stdout)
        self.assertEqual(plan["total_solves"], 828)
        self.assertEqual([name for name, _ in plan["groups"]], ["main", "public", "scaling"])


if __name__ == "__main__":
    unittest.main()
