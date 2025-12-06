import os

import h5py
import jax.numpy as jnp
import numpy as np
from fftlog import config
from fftlog.fftlog import FFTLog
from jax import jit, vmap
from scipy.constants import speed_of_light

from . import cl_kdep, coeff

config.set_jax_enabled(True)


class ClComp(object):
    """
    A class to compute the angular power spectrum given the power spectrum and the window function.

    Attributes:
        z1 (np.ndarray): Redshift endpoints of the window function of the first contribution.
        z2 (np.ndarray): Redshift endpoints of the window function of the second contribution.
        l (jnp.ndarray): Values of multipole moment `l` to compute.
        k (jnp.ndarray): Values of wave number `k` to integrate over.
        contr1 (str, optional): First probe for the computation of the power spectrum.
            Can be one of 'g', 'mag', 'RSD', 'f_NL', 'g,mag', 'g,RSD', 'g,mag,RSD', 'g,f_NL', 'wl', 'CMB_k', 'CMB_T', 'intjl', 'intjl/k^2' or 'intddjl'. Defaults to 'g'.
        contr2 (str, optional): Second probe for the computation of the power spectrum.
            Can be one of 'g', 'mag', 'RSD', 'f_NL', 'g,mag', 'g,RSD', 'g,mag,RSD', 'g,f_NL', 'wl', 'CMB_k', or 'CMB_T', 'intjl', 'intjl/k^2' or 'intddjl'. Defaults to 'g'.
        jit (bool, optional): Enables just-in-time compilation. Defaults to False.
        N (int, optional): Maximum number of points for the FFTLog of the window function. Defaults to 512.
        bias (float, optional): Bias for the FFTLog. Must be between 0 and 1. Defaults to 0.01.
        wind (float, optional): Smoothing parameter for the window function. Defaults to 0.2.
        N_interp (int, optional): Number of points for interpolation along `k`. Defaults to 100.
        path (str, optional): Path to save the `g_r` files. Defaults to the current working directory.
    """

    def __init__(self, **kwargs):
        self.c = dict(**kwargs)

        self.set_config("contr1", "g")
        self.set_config("contr2", "g")
        self.set_config("N", 512)
        self.set_config("bias", 0.1)
        self.set_config("wind", 0.2)

        self.set_config("jit", False)
        self.set_config("N_interp", 200)
        self.set_config("path", os.getcwd())
        self.set_config("emu", "jax-cosmo")

        # computes integration boundaries
        c = speed_of_light / 1000
        if self.c["z1"][0] == 0.0:
            self.c["xmin1"] = 10e-4
        else:
            self.c["xmin1"] = 10 ** int(np.log10(self.c["z1"][0] * c / (0.7 * 100)) - 3)
        self.c["xmax1"] = 10 ** int(np.log10(self.c["z1"][-1] * c / (0.7 * 100)) + 2)

        if self.c["z2"][0] == 0.0:
            self.c["xmin2"] = 10e-4
        else:
            self.c["xmin2"] = 10 ** int(np.log10(self.c["z2"][0] * c / (0.7 * 100)) - 3)
        self.c["xmax2"] = 10 ** int(np.log10(self.c["z2"][-1] * c / (0.7 * 100)) + 2)

        # for CMB contributions, z_* = 1100
        if self.c["contr1"] == "CMB_k":
            self.c["xmin1"] = 10e-4
            self.c["xmax1"] = 10 ** int(np.log10(1100 * c / (0.7 * 100)) + 2)

        if self.c["contr2"] == "CMB_k":
            self.c["xmin2"] = 10e-4
            self.c["xmax2"] = 10 ** int(np.log10(1100 * c / (0.7 * 100)) + 2)

        # precomputes the gamma prefactors of the chi integrals
        self.c["g_r_1"] = self.c_int(self.c["xmin1"], self.c["xmax1"])
        self.c["g_r_2"] = self.c_int(self.c["xmin2"], self.c["xmax2"])

    def set_config(self, option, default):
        if option not in self.c:
            self.c[option] = default

    def c_int(self, xmin, xmax):
        # checks that directory exists
        folder_name = self.c["path"] + "g_r"
        os.makedirs(folder_name, exist_ok=True)

        fftsettings = dict(
            Nmax=self.c["N"],
            xmin=xmin,
            xmax=xmax,
            bias=self.c["bias"],
            window=self.c["wind"],
        )
        fft = FFTLog(**fftsettings)
        power_fft = fft.Pow

        file_path = f"{folder_name}/g_r_N_{self.c['N']}_xmin_{xmin}_xmax_{xmax}.h5"

        # checks that I_l file already exists
        if not os.path.isfile(file_path):
            coeff.main(self, power_fft, xmin, xmax)

        # checks that parameters are the same
        else:
            with h5py.File(
                file_path,
                "r",
            ) as file:
                l_max = file["g_r_jl"].attrs["l_max"]
                bias = file["g_r_jl"].attrs["bias"]
                w = file["g_r_jl"].attrs["window"]
            if l_max < self.c["l"][-1] or bias != self.c["bias"] or w != self.c["wind"]:
                coeff.main(self, power_fft, xmin, xmax)

        with h5py.File(
            file_path,
            "r",
        ) as file:
            g_r_jl = jnp.array(
                [file["g_r_jl"][ll] for ll in self.c["l"]], dtype="complex64"
            )
            g_r_k = jnp.array(
                [file["g_r_k"][ll] for ll in self.c["l"]], dtype="complex64"
            )
            g_r_ddjl = jnp.array(
                [file["g_r_ddjl"][ll] for ll in self.c["l"]], dtype="complex64"
            )

        return jnp.array([g_r_jl, g_r_k, g_r_ddjl])

    # Window functions
    def w_g(self, n, H, D):
        c = speed_of_light / 1000
        return jnp.einsum("c, ck -> ck", n[:, 1] * H / c, D)  # Mpc^-1

    def w_RSD(self, n, H, f, D):
        c = speed_of_light / 1000
        return jnp.einsum("c, ck -> ck", n[:, 1] * H * f / c, D)  # Mpc^-1

    def w_k(self, n, chis, D):
        chis = jnp.where(chis == 0.0, 1e-4, chis)
        c = speed_of_light / 1000
        a = jnp.array(1 / (1 + n[:, 0]))
        n_mod = jnp.array(
            [
                jnp.append(jnp.zeros(len(n[:, 1]) - len(n[i:, 1])), n[i:, 1])
                for i in range(len(n[:, 1]))
            ]
        )
        I = vmap(
            lambda i: jnp.trapezoid(n_mod[i] * (chis - chis[i]) / chis, x=n[:, 0])
        )(jnp.arange(len(n[:, 0]), dtype=int))
        return jnp.einsum("c, ck -> ck", 3 / (2 * c**2) * I * chis / a, D)

    def w_fnl(self, n, H, T, D):
        c = speed_of_light / 1000
        n_bar = jnp.trapezoid(n[:, 1], x=n[:, 0])
        return (
            2 / 3 * jnp.einsum("c, ck -> ck", 1 / n_bar * n[:, 1] * H / c, 1 / T)
        )  # Mpc^-1

    def w_wl(self, n, chis, H, H0, O_m, A_IA, D):
        chis = jnp.where(chis == 0.0, 1e-4, chis)
        c = speed_of_light / 1000
        a = jnp.array(1 / (1 + n[:, 0]))
        n_mod = jnp.array(
            [
                jnp.append(jnp.zeros(len(n[:, 1]) - len(n[i:, 1])), n[i:, 1])
                for i in range(len(n[:, 1]))
            ]
        )
        I = vmap(
            lambda i: jnp.trapezoid(n_mod[i] * (chis - chis[i]) / chis, x=n[:, 0])
        )(jnp.arange(len(n[:, 0]), dtype=int))
        return jnp.einsum(
            "c, ck -> ck",
            3 * O_m * H0**2 / (2 * c**2) * I * chis / a + n[:, 1] * H / c * A_IA,
            D,
        )

    def w_CMB_k(self, n, chis, D, chi_ls):
        c = speed_of_light / 1000
        a = jnp.array(1 / (1 + n[:, 0]))
        return jnp.einsum(
            "c, ck -> ck", 3 / (2 * c**2) * (chi_ls - chis) * chis / (a * chi_ls), D
        )

    def w_CMB_T(self, chis, H, f, D):
        c = speed_of_light / 1000
        return jnp.einsum(
            "c, k -> ck", 3 / c**3 * H * (1 - f) * D * chis**2, 1 / self.c["k"] ** 2
        )

    # returns array of window functions
    def window(
        self, contr, n, chis, H, H0, O_m, f, T, D, C_g, b_g, b_fNL, f_NL, A_IA, chi_ls
    ):
        if (
            contr != "intjl" and contr != "intjl/k^2" and contr != "intddjl"
        ):  # and contr != 'intdjl'
            I_n = jnp.trapezoid(n[:, 1], x=n[:, 0])
            n = n.at[:, 1].set(n[:, 1] / I_n)

        # galaxy clustering
        if contr == "g":
            return self.w_g(n, H, D) * b_g
        elif contr == "mag":
            return self.w_k(n, chis, D) * C_g * O_m * H0**2
        elif contr == "RSD":
            return self.w_RSD(n, H, f, D)
        elif contr == "f_NL" or contr == "f_NL_mod":
            return self.w_fnl(n, H, T, D) * b_fNL * f_NL
        elif contr == "g,mag":
            return jnp.array(
                [self.w_g(n, H, D) * b_g, self.w_k(n, chis, D) * C_g * O_m * H0**2]
            )
        elif contr == "g,RSD":
            return jnp.array([self.w_g(n, H, D) * b_g, self.w_RSD(n, H, f, D)])
        elif contr == "g,f_NL":
            return self.w_g(n, H, D) * b_g + self.w_fnl(n, H, T, D) * b_fNL * f_NL
        elif contr == "g,mag,RSD":
            return jnp.array(
                [
                    self.w_g(n, H, D) * b_g,
                    self.w_k(n, chis, D) * C_g * O_m * H0**2,
                    self.w_RSD(n, H, f, D),
                ]
            )
        # weak lensing
        elif contr == "wl":
            return self.w_wl(n, chis, H, H0, O_m, A_IA, D)
        # CMB
        elif contr == "CMB_k":
            return self.w_CMB_k(n, chis, D, chi_ls) * O_m * H0**2
        elif contr == "CMB_T":
            T_CMB = 2.7255
            return self.w_CMB_T(chis, H, f, D) * T_CMB * O_m * H0**2
        # contributions with kernels input
        elif (
            contr == "intjl" or contr == "intjl/k^2" or contr == "intddjl"
        ):  # or contr == 'intdjl'
            return n[..., jnp.newaxis] * D
        else:
            raise TypeError("Probe not in the list")

    def C_l(
        self,
        n1,
        n2,
        chis1,
        chis2,
        D1,
        D2,
        P,
        H1=1.0,
        H2=1.0,
        H0=1.0,
        O_m=1.0,
        f1=1.0,
        f2=1.0,
        T1=1.0,
        T2=1.0,
        C_g1=1.0,
        C_g2=1.0,
        b_g1=1.0,
        b_g2=1.0,
        b_fNL1=1.0,
        b_fNL2=1.0,
        f_NL1=1.0,
        f_NL2=1.0,
        A_IA1=0.0,
        A_IA2=0.0,
        chi_ls=13873.39,
    ):
        """
        Compute the angular power spectrum assuming a scale-dependent growth factor.

        Args:
            n1 (np.ndarray): Redshift distribution for the first probe. A 2D array where `n1[:,0]` is redshift `z` and `n1[:,1]` is the probability density `p(z)`.
            n2 (np.ndarray): Redshift distribution for the second probe, same format as `n1`.
            chis1 (np.ndarray): Comoving distance χ(z) evaluated at `n1[:,0]`.
            chis2 (np.ndarray): Comoving distance χ(z) evaluated at `n2[:,0]`.
            D1 (np.ndarray): Growth factor D(z, k) for the first probe.
            D2 (np.ndarray): Growth factor D(z, k) for the second probe.
            P (np.ndarray): Power spectrum P(k, z0) at a fiducial redshift z0.
            H1 (np.ndarray): Hubble parameter H(z) evaluated at `n1[:,0]` (needed for 'g' contribution).
            H2 (np.ndarray): Hubble parameter H(z) evaluated at `n2[:,0]` (needed for 'g' contribution).
            H0 (float, optional): Hubble parameter at z=0. Required for 'mag' and 'wl' contributions. Defaults to 1.
            O_m (float, optional): Matter density parameter Ω_m. Required for 'mag' and 'wl' contributions. Defaults to 1.
            f1 (np.ndarray, optional): Logarithmic growth rate f(z) at `n1[:,0]`. Required for 'RSD'. Defaults to 1.
            f2 (np.ndarray, optional): Logarithmic growth rate f(z) at `n2[:,0]`. Required for 'RSD'. Defaults to 1.
            T1 (np.ndarray, optional): Matter transfer function T(z,k) for the first probe. Required for 'f_NL'. Defaults to 1.
            T2 (np.ndarray, optional): Matter transfer function T(z,k) for the second probe. Required for 'f_NL'. Defaults to 1.
            C_g1 (float, optional): Magnification bias coefficient for the first probe. Defaults to 1.
            C_g2 (float, optional): Magnification bias coefficient for the second probe. Defaults to 1.
            b_g1 (float, optional): Galaxy bias for the first probe. Defaults to 1.
            b_g2 (float, optional): Galaxy bias for the second probe. Defaults to 1.
            b_fNL1 (float, optional): Non-Gaussian bias parameter for the first probe, \(b_1^{f_{NL}}\). Defaults to 1.
            b_fNL2 (float, optional): Non-Gaussian bias parameter for the second probe, \(b_1^{f_{NL}}\). Defaults to 1.
            f_NL1 (float, optional): Primordial non-Gaussianity parameter for the first probe. Defaults to 1.
            f_NL2 (float, optional): Primordial non-Gaussianity parameter for the second probe. Defaults to 1.
            A_IA1 (np.ndarray, optional): Intrinsic alignment amplitude for the first probe, evaluated at `n1[:,0]`. Defaults to 0.
            A_IA2 (np.ndarray, optional): Intrinsic alignment amplitude for the second probe, evaluated at `n2[:,0]`. Defaults to 0.

        Returns:
            np.ndarray: Angular power spectrum \(C_\ell\) as a 1D array.
        """

        function = (
            cl_kdep.C_dep if not self.c["jit"] else jit(cl_kdep.C_dep, static_argnums=0)
        )

        return function(
            self,
            n1,
            n2,
            P,
            chis1,
            chis2,
            D1,
            D2,
            H1,
            H2,
            H0,
            O_m,
            f1,
            f2,
            T1,
            T2,
            C_g1,
            C_g2,
            b_g1,
            b_g2,
            b_fNL1,
            b_fNL2,
            f_NL1,
            f_NL2,
            A_IA1,
            A_IA2,
            chi_ls,
        )
