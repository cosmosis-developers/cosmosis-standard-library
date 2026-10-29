"""Compare the proposed grid interpolation against CLASS scalar outputs."""
import json
import time
import numpy as np
from classy import Class

c = Class()
c.set(dict(output='mPk', h=0.7, omega_b=0.0224, omega_cdm=0.12,
           A_s=2.1e-9, n_s=0.965, z_max_pk=3., YHe=0.245,
           recombination='RECFAST',
           **{'P_k_max_h/Mpc': 10., 'non linear': 'halofit'}))
try:
    c.compute()
    k = np.logspace(-4, np.log10(7.), 100)
    z = np.arange(0., 3.01, 0.01)
    for nonlinear in (False, True):
        scalar = c.pk if nonlinear else c.pk_lin
        old_times, new_times = [], []
        for repeat in range(3):
            start = time.perf_counter()
            old = np.array([[scalar(ki, zi) for zi in z] for ki in k])
            old_times.append(time.perf_counter() - start)
            start = time.perf_counter()
            karr = k[:, None, None] * np.ones((1, z.size, 1))
            new = (c.get_pk(karr, z, k.size, z.size, 1) if nonlinear
                   else c.get_pk_lin(karr, z, k.size, z.size, 1))[:, :, 0]
            new_times.append(time.perf_counter() - start)
        error = np.abs(new-old)/np.maximum(np.abs(old), 1e-30)
        print(json.dumps(dict(nonlinear=nonlinear, max_relative_error=float(error.max()),
              median_old_seconds=float(np.median(old_times)), median_new_seconds=float(np.median(new_times)),
              speedup=float(np.median(old_times)/np.median(new_times)),
              finite=bool(np.isfinite(new).all()), positive=bool((new>0).all()),
              inside_bounds=True)), flush=True)
finally:
    c.struct_cleanup()
    c.empty()
