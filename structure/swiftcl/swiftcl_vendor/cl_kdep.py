import jax.numpy as jnp
from jax import vmap

from . import interpolate


def g_contr(self, W, g_r, chis, xmin, xmax):
    par = interpolate.interp(
        l=self.c["l"],
        N=self.c["N"],
        x=chis,
        xmin=xmin,
        xmax=xmax,
        f=W,
        a=self.c["k"],
        b=self.c["bias"],
        g_r=g_r,
        N_interp=self.c["N_interp"],
    )
    return par.int_jl()


def RSD_contr(self, W, g_r, chis, xmin, xmax):
    par_RSD = interpolate.interp(
        l=self.c["l"],
        N=self.c["N"],
        x=chis,
        xmin=xmin,
        xmax=xmax,
        f=W,
        a=self.c["k"],
        b=self.c["bias"],
        g_r=g_r,
        N_interp=self.c["N_interp"],
    )
    return par_RSD.int_ddjl()


def wl_contr(self, W, g_r, chis, xmin, xmax):
    par_wl = interpolate.interp(
        l=self.c["l"],
        N=self.c["N"],
        x=chis,
        xmin=xmin,
        xmax=xmax,
        f=W,
        a=self.c["k"],
        b=self.c["bias"],
        g_r=g_r,
        N_interp=self.c["N_interp"],
    )
    return par_wl.int_k()


def I(self, contr, W, g_r, chis, xmin, xmax):
    if contr == "g":
        return g_contr(self, W, g_r[0], chis, xmin, xmax)
    elif contr == "f_NL":
        return g_contr(self, W, g_r[0], chis, xmin, xmax)
    elif contr == "f_NL_mod":
        return jnp.einsum(
            "k, ck -> ck",
            1 / self.c["k"] ** 2,
            wl_contr(self, W, g_r[1], chis, xmin, xmax),
        )
    elif contr == "g,f_NL":
        return g_contr(self, W, g_r[0], chis, xmin, xmax)
    elif contr == "mag":
        pref_mag = jnp.outer(self.c["l"] * (self.c["l"] + 1), 1 / self.c["k"] ** 2)
        return pref_mag * wl_contr(self, W, g_r[1], chis, xmin, xmax)
    elif contr == "RSD":
        return -RSD_contr(self, W, g_r[2], chis, xmin, xmax)
    elif contr == "g,mag":
        pref_mag = jnp.outer(self.c["l"] * (self.c["l"] + 1), 1 / self.c["k"] ** 2)
        return g_contr(self, W[0], g_r[0], chis, xmin, xmax) + pref_mag * wl_contr(
            self, W[1], g_r[1], chis, xmin, xmax
        )
    elif contr == "g,RSD":
        return g_contr(self, W[0], g_r[0], chis, xmin, xmax) - RSD_contr(
            self, W[1], g_r[2], chis, xmin, xmax
        )
    elif contr == "g,mag,RSD":
        pref_mag = jnp.outer(self.c["l"] * (self.c["l"] + 1), 1 / self.c["k"] ** 2)
        return (
            g_contr(self, W[0], g_r[0], chis, xmin, xmax)
            + pref_mag * wl_contr(self, W[1], g_r[1], chis, xmin, xmax)
            - RSD_contr(self, W[2], g_r[2], chis, xmin, xmax)
        )
    elif contr == "wl":
        pref_wl = jnp.outer(
            jnp.sqrt(
                (self.c["l"] + 2) * (self.c["l"] + 1) * self.c["l"] * (self.c["l"] - 1)
            ),
            1 / self.c["k"] ** 2,
        )
        return pref_wl * wl_contr(self, W, g_r[1], chis, xmin, xmax)
    elif contr == "CMB_k":
        pref_CMB = jnp.outer((self.c["l"] + 1) * self.c["l"], 1 / self.c["k"] ** 2)
        return pref_CMB * wl_contr(self, W, g_r[1], chis, xmin, xmax)
    elif contr == "CMB_T":
        return wl_contr(self, W, g_r[1], chis, xmin, xmax)

    elif contr == "intjl":
        return g_contr(self, W, g_r[0], chis, xmin, xmax)
    ### TO change !!!
    elif contr == "intjl/k^2":
        pref_wl = jnp.outer(
            jnp.sqrt(
                (self.c["l"] + 2) * (self.c["l"] + 1) * self.c["l"] * (self.c["l"] - 1)
            ),
            1 / self.c["k"] ** 2,
        )
        return pref_wl * wl_contr(self, W, g_r[1], chis, xmin, xmax)
    elif contr == "intddjl":
        return RSD_contr(self, W, g_r[2], chis, xmin, xmax)
    # elif contr == 'intdjl':
    # return djl_contr(self, W, g_r[3], chis, xmin, xmax)


def C_dep(
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
):
    W1 = self.window(
        self.c["contr1"],
        n1,
        chis1,
        H1,
        H0,
        O_m,
        f1,
        T1,
        D1,
        C_g1,
        b_g1,
        b_fNL1,
        f_NL1,
        A_IA1,
        chi_ls,
    )
    W2 = self.window(
        self.c["contr2"],
        n2,
        chis2,
        H2,
        H0,
        O_m,
        f2,
        T2,
        D2,
        C_g2,
        b_g2,
        b_fNL2,
        f_NL2,
        A_IA2,
        chi_ls,
    )

    int_1 = I(
        self,
        self.c["contr1"],
        W1,
        self.c["g_r_1"],
        chis1,
        self.c["xmin1"],
        self.c["xmax1"],
    )
    int_2 = I(
        self,
        self.c["contr2"],
        W2,
        self.c["g_r_2"],
        chis2,
        self.c["xmin2"],
        self.c["xmax2"],
    )

    C_l = (
        2.0
        / jnp.pi
        * vmap(
            lambda i: jnp.trapezoid(
                self.c["k"] ** 2 * int_1[i, :] * int_2[i, :] * P, x=self.c["k"]
            )
        )(jnp.arange(len(self.c["l"]), dtype=int))
    )
    return C_l
