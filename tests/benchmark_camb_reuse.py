"""Reproducible benchmark for exact CAMB result reuse."""

import time

import numpy as np

from tests.unit_tests.test_camb import _camb_power_block, build_camb_module


def main():
    fresh = build_camb_module(mode="power", reuse_computation=False, nk=40, nz=12)
    reused = build_camb_module(mode="power", reuse_computation=True,
                               reuse_cache_size=4, nk=40, nz=12)
    reference_block = _camb_power_block()
    reuse_block = _camb_power_block()

    start = time.perf_counter()
    assert fresh.execute(reference_block) == 0
    fresh_time = time.perf_counter() - start

    assert reused.execute(_camb_power_block()) == 0
    start = time.perf_counter()
    assert reused.execute(reuse_block) == 0
    reuse_time = time.perf_counter() - start

    reference = reference_block["matter_power_lin", "p_k"]
    result = reuse_block["matter_power_lin", "p_k"]
    epsilon = np.max(np.abs(result - reference) /
                     np.maximum(np.abs(reference), 1e-30))
    speedup = fresh_time / reuse_time
    print(f"fresh_seconds={fresh_time:.6f}")
    print(f"reuse_seconds={reuse_time:.6f}")
    print(f"speedup={speedup:.3f}")
    print(f"epsilon_max={epsilon:.3e}")
    assert np.isfinite(epsilon)
    assert epsilon <= 1e-12


if __name__ == "__main__":
    main()
