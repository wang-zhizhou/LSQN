"""Native preparation is explicit; ABI mismatch and missing builds are errors."""
import ctypes
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from lsqn import _native


class NativePreparationTests(unittest.TestCase):
    def setUp(self):
        self.previous = _native._STATE
        _native._STATE = None

    def tearDown(self):
        _native._STATE = self.previous

    def test_missing_build_never_invokes_compiler(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"LSQN_NATIVE_CACHE": directory}):
                with patch.object(_native, "_compiler", side_effect=AssertionError("implicit compiler")):
                    with self.assertRaisesRegex(_native.NativeBackendError, "not prepared"):
                        _native.get_library()

    def test_invalid_dimensions_rejected(self):
        for dimension in (-1, 2**31, 1.5):
            with self.assertRaises(_native.NativeBackendError):
                _native.validate_dimensions(dimension)

    def test_disagreeing_blas_integer_declaration_rejected(self):
        for bits, description in ((32, "OpenBLAS test USE64BITINT"), (64, "OpenBLAS test")):
            with self.assertRaisesRegex(_native.NativeBackendError, "integer configuration disagree"):
                _native._validate_abi_config(bits, description)

    @unittest.skipUnless(any(shutil.which(x) for x in ("cc", "clang", "gcc")), "C compiler unavailable")
    def test_lp64_kernel_build_with_actual_blas(self):
        # The archived macOS environment also provides an independent LP64
        # OpenBLAS through SciPy. Use it only as a test fixture: production
        # discovery still exclusively inspects NumPy's own extension.
        import scipy.linalg._fblas
        try:
            blas = _native._detect_openblas(ctypes.CDLL(scipy.linalg._fblas.__file__))
        except _native.NativeBackendError as exc:
            self.skipTest("A second supported BLAS fixture is unavailable: " + str(exc))
        if blas["bits"] != 32:
            self.skipTest("This environment has no LP64 BLAS fixture")
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"LSQN_NATIVE_CACHE": directory}):
                with patch.object(_native, "_discover_blas", return_value=blas):
                    metadata = _native.prepare()
                    self.assertEqual(metadata["identity"]["blas_integer_bits"], 32)
                    self.assertTrue(_native.selftest()["passed"])

    @unittest.skipUnless(any(shutil.which(x) for x in ("cc", "clang", "gcc")), "C compiler unavailable")
    def test_prepare_identities_reload_and_binary_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"LSQN_NATIVE_CACHE": directory}):
                metadata = _native.prepare()
                self.assertTrue(metadata["prepared"])
                self.assertIn(metadata["identity"]["blas_integer_bits"], (32, 64))
                self.assertTrue(_native.selftest()["passed"])
                binary = Path(metadata["binary"])
                _native._STATE = None
                with patch.object(_native, "_compiler", side_effect=AssertionError("implicit compiler")):
                    self.assertIsNotNone(_native.get_library())
                self.assertEqual(_native.get_library().lsqn_blas_abi_bits(), metadata["identity"]["blas_integer_bits"])
                _native._STATE = None
                manifest = binary.parent / "build.json"
                text = manifest.read_text()
                manifest.write_text(text.replace('"binary_sha256": "', '"binary_sha256": "tampered'))
                with self.assertRaisesRegex(_native.NativeBackendError, "does not match"):
                    _native.get_library()


if __name__ == "__main__":
    unittest.main()
