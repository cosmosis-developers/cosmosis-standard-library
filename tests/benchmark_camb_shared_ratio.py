"""Experimental native-grid ratio sharing; does not alter production execution."""
import time
import numpy as np
import camb
from tests.unit_tests.test_camb import build_camb_module, _camb_power_block
from boltzmann.camb.camb_interface import extract_camb_params
from boltzmann.camb.camb_interface import recompute_for_sigma8


class SharedRatio:
    def __init__(self, result):
        self.result = result
        self.ratio = None
        self.grid = None

    def __getattr__(self, name):
        return getattr(self.result, name)

    def get_linear_matter_power_spectrum(self, *args, **kwargs):
        if not kwargs.get("nonlinear", False):
            return self.result.get_linear_matter_power_spectrum(*args, **kwargs)
        linear_args = dict(kwargs, nonlinear=False)
        k, z, linear = self.result.get_linear_matter_power_spectrum(*args, **linear_args)
        if self.ratio is None:
            kn, zn, nonlinear = self.result.get_linear_matter_power_spectrum(*args, **kwargs)
            np.testing.assert_array_equal(k, kn)
            np.testing.assert_array_equal(z, zn)
            if not np.all(np.isfinite(linear)) or np.any(linear <= 0):
                raise ValueError("Ratio sharing requires finite positive reference power")
            self.ratio = nonlinear / linear
            self.grid = (k.copy(), z.copy())
            return kn, zn, nonlinear
        np.testing.assert_array_equal(k, self.grid[0])
        np.testing.assert_array_equal(z, self.grid[1])
        return k, z, linear * self.ratio

    def get_matter_power_interpolator(self, **kwargs):
        return camb.results.CAMBdata.get_matter_power_interpolator(self, **kwargs)


if __name__ == "__main__":
    module = build_camb_module(mode="power", nz=40, nk=100)
    config, extra = module.data
    for mass, extrapolate, sigma8 in ((0.06, False, None), (0.15, True, None),
                                    (0.30, True, 0.8), (0.06, False, 0.85)):
        block = _camb_power_block()
        block["cosmological_parameters", "mnu"] = mass
        if sigma8 is not None:
            block["cosmological_parameters", "sigma_8"] = sigma8
        result = camb.get_results(extract_camb_params(block, config, extra))
        if sigma8 is not None:
            result = recompute_for_sigma8(result, block, config, extra)
            assert abs(result.get_sigma8_0()/sigma8-1) <= 1e-6
        timings = {False: [], True: []}
        errors = []
        for repeat in range(4):
            powers = {}
            for sharing in ((False, True) if repeat % 2 == 0 else (True, False)):
                obj = SharedRatio(result) if sharing else result
                start = time.perf_counter()
                outputs = []
                for variable in ("delta_tot", "delta_nonu", "delta_cdm"):
                    p = obj.get_matter_power_interpolator(
                        nonlinear=True, var1=variable, var2=variable,
                        extrap_kmax=20.0 if extrapolate else None)
                    outputs.append(p.P(np.linspace(0.03, 2.97, 20),
                                       np.geomspace(1e-4, 15 if extrapolate else 1, 80)))
                timings[sharing].append(time.perf_counter()-start)
                powers[sharing] = outputs
            epsilon = max(float(np.max(abs(a-b)/np.maximum(abs(a), 1e-30)))
                          for a, b in zip(powers[False], powers[True]))
            assert all(np.all(np.isfinite(p)) and np.all(p > 0) for p in powers[True])
            assert epsilon <= 1e-12
            errors.append(epsilon)
        print("RESULT", mass, "extrapolate", extrapolate, "sigma8", sigma8,
              "median", [float(np.median(timings[k])) for k in (False, True)],
              "std", [float(np.std(timings[k], ddof=1)) for k in (False, True)],
              "epsilon", max(errors), flush=True)
