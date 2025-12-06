import h5py
import jax.numpy as jnp
import numpy as np
from mpmath import gamma


def main(self, power_fft, xmin, xmax):
    print(
        "First initialisation: computing the gamma coefficients, this might take a few minutes..."
    )

    ll = np.arange(0, self.c["l"][-1] + 1, dtype=int)
    pow_fft = np.array(power_fft)

    g_r_jl = jnp.array(
        [
            [gamma(0.5 * (1 + l + p)) / gamma(0.5 * (2 + l - p)) for p in pow_fft]
            for l in ll
        ],
        dtype="complex64",
    )
    g_r_k = jnp.array(
        [
            [gamma(0.5 * (1 + l + p)) / gamma(0.5 * (2 + l - p)) for p in (pow_fft - 2)]
            for l in ll
        ],
        dtype="complex64",
    )
    g_r_ddjl = jnp.array(
        [
            [
                gamma(0.5 * (-1 + l + p)) / gamma(0.5 * (4 + l - p))
                for p in (pow_fft + 0.91)
            ]
            for l in ll
        ],
        dtype="complex64",
    )

    hf = h5py.File(
        f"{self.c['path']}g_r/g_r_N_{self.c['N']}_xmin_{xmin}_xmax_{xmax}.h5", "w"
    )
    dset = hf.create_dataset("g_r_jl", data=g_r_jl)
    hf.create_dataset("g_r_k", data=g_r_k)
    hf.create_dataset("g_r_ddjl", data=g_r_ddjl)
    dset.attrs["l_min"] = self.c["l"][0]
    dset.attrs["l_max"] = self.c["l"][-1]
    dset.attrs["x_min"] = xmin
    dset.attrs["x_max"] = xmax
    dset.attrs["bias"] = self.c["bias"]
    dset.attrs["window"] = self.c["wind"]
    dset.attrs["N"] = self.c["N"]
    hf.close()
