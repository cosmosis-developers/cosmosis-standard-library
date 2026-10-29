"""Run directly with a CLASS 3.2.5 + CosmoSIS Python environment."""
import importlib.util
import json
from pathlib import Path
import subprocess
import time
import types
import unittest

import numpy as np
import classy
from cosmosis.datablock import DataBlock

ROOT = Path(__file__).resolve().parents[1]
PATH = 'boltzmann/class/class_interface.py'
spec = importlib.util.spec_from_file_location('batch_interface', ROOT / PATH)
batch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(batch)
scalar = types.ModuleType('scalar_interface')
scalar.__file__ = str(ROOT / PATH)
# Pin the actual upstream implementation before this PR.
exec(subprocess.check_output(['git', 'show', '9e3dd611fc20bf39489d43bf4afeb503d26b4e79:' + PATH], cwd=ROOT), scalar.__dict__)


def run(interface, mass):
    block = DataBlock()
    for name, value in dict(h0=.7, A_s=2.1e-9, n_s=.965,
                            ombh2=.0224, omch2=.12, tau=.054,
                            mnu=mass, num_massive_neutrinos=1).items():
        block['cosmological_parameters', name] = value
    config = dict(cosmo=classy.Class(), kmax=10., nk=100, zmax=3.,
                  cmb=False, lensing=False, mpk=True, background_only=False,
                  save_matter_power_lin=True, save_cdm_baryon_power_lin=True,
                  debug=True, class_YHe=.245, class_recombination='RECFAST',
                  **{'class_non linear': 'halofit'})
    try:
        start = time.perf_counter()
        status = interface.execute(block, config)
        elapsed = time.perf_counter()-start
        assert status == 0
        return block, elapsed
    finally:
        interface.cleanup(config)


class ClassBatchRegression(unittest.TestCase):
    def test_outputs_and_end_to_end_timing(self):
        # Warm up both paths; interleave order to reduce order-dependent bias.
        run(scalar, .06)
        run(batch, .06)
        old_times, new_times = [], []
        for mass in (.06, .15, .30):
            for repeat in range(4):
                results = {}
                for name, interface in ((('old', scalar), ('new', batch)) if repeat % 2 == 0
                                        else (('new', batch), ('old', scalar))):
                    results[name] = run(interface, mass)
                old, ot = results['old']; new, nt = results['new']
                old_times.append(ot); new_times.append(nt)
                self.assertEqual(set(old.keys()), set(new.keys()))
                for key in old.keys():
                    if np.asarray(old[key]).dtype.kind in 'US':
                        np.testing.assert_equal(new[key], old[key])
                        continue
                    np.testing.assert_allclose(new[key], old[key], rtol=1e-10, atol=0,
                                               err_msg=str(key))
                for section in ('matter_power_lin', 'cdm_baryon_power_lin', 'matter_power_nl'):
                    a, b = new[section, 'p_k'], old[section, 'p_k']
                    self.assertTrue(np.isfinite(a).all() and (a > 0).all())
                    error = np.max(np.abs(a-b)/np.maximum(np.abs(b), 1e-30))
                    self.assertLessEqual(error, 1e-10)
                    print(json.dumps(dict(mass_eV=mass, repeat=repeat, section=section,
                                          max_relative_error=float(error))))
        ot, nt = float(np.median(old_times)), float(np.median(new_times))
        print('TIMING', json.dumps(dict(pairs=len(old_times), old_median_s=ot,
              new_median_s=nt, speedup=ot/nt, reduction_percent=100*(1-nt/ot),
              backend=classy.__file__)))


if __name__ == '__main__':
    unittest.main()
