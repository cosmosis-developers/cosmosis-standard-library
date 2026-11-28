"""Preserve failed FITPACK inputs, including exceptions caught by samplers."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest


failure_count = 0


def capture_failed_interpolation(frame, event, result):
    if event != "return" or result is not None:
        return
    if frame.f_code.co_name != "_curfit":
        return
    if not frame.f_code.co_filename.endswith("scipy/interpolate/_fitpack2.py"):
        return

    # Profiling observes the failed return without replacing scientific routines.
    global failure_count
    failure_count += 1
    arrays = {}
    details = {}
    current = frame
    try:
        while current is not None:
            name = current.f_code.co_name
            values = current.f_locals
            if name in ("_curfit", "__init__", "set_wofchi_splines", "load_kernels"):
                details.setdefault(name, {})
                for key in ("i", "sample_name", "kernel_type", "quick_norm", "norm"):
                    value = values.get(key)
                    if isinstance(value, (str, int, float, bool)):
                        details[name][key] = value
                for key, value in values.items():
                    if isinstance(value, np.ndarray) and value.dtype.kind in "fbiu":
                        arrays[f"{name}_{key}"] = value.copy()
                if name == "__init__" and "kernel.py" in current.f_code.co_filename:
                    kernel = values["self"]
                    arrays["kernel_x"] = kernel.x.copy()
                    arrays["kernel_y"] = kernel.y.copy()
                if name == "set_wofchi_splines":
                    kernel = values["nchi_spline"]
                    arrays["nchi_x"] = kernel.x.copy()
                    arrays["nchi_y"] = kernel.y.copy()
            current = current.f_back
        directory = Path("output/interpolation-diagnostics")
        directory.mkdir(parents=True, exist_ok=True)
        filename = directory / f"failure-{failure_count}.npz"
        np.savez_compressed(filename, **arrays)
        details["arrays"] = {
            key: {"shape": list(value.shape), "finite": bool(np.isfinite(value).all()),
                  "nonfinite_count": int(np.count_nonzero(~np.isfinite(value)))}
            for key, value in arrays.items()
        }
        details["sha256"] = hashlib.sha256(filename.read_bytes()).hexdigest()
        print("INTERPOLATION_FAILURE", filename, json.dumps(details), flush=True)
    except Exception as error:
        print("INTERPOLATION_DIAGNOSTIC_ERROR", repr(error), flush=True)


if __name__ == "__main__":
    sys.path.insert(0, str(Path.cwd()))
    previous = sys.getprofile()
    sys.setprofile(capture_failed_interpolation)
    try:
        status = pytest.main(sys.argv[1:])
    finally:
        sys.setprofile(previous)
    raise SystemExit(status)
