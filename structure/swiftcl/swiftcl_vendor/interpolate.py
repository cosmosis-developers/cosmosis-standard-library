import jax
import jax.numpy as jnp
from fftlog.fftlog import FFTLog
from interpax import interp1d


class interp(object):
    """
    A class to compute integrals of Bessel functions multiplied by a function using FFTLog-based techniques.

    This class supports evaluation of three integral variants:
    - Standard spherical Bessel integrals (`int_jl`)
    - Integrals with additional `k^-2` scaling (`int_k`)
    - Second-derivative Bessel integrals (`int_ddjl`)

    Args:
        l (array_like): Values of l to compute.
        N (int): Maximum number of points for the FFTLog of the window function.
        x (array_like): Points where the input function is evaluated.
        f (array_like): 2D array of shape (len(x), len(a)) representing the function multiplying the Bessel function.
        a (array_like): Values multiplying the Bessel function argument, i.e., in j_l(ax).
        b (float): Bias for the FFTLog.
        g_r (array_like): Precomputed Gamma coefficients.
        xmin (float): Minimum value of `x` for FFTLog.
        xmax (float): Maximum value of `x` for FFTLog.
        N_interp (int): Number of points to interpolate across `a`.

    Attributes:
        l (ndarray): Spherical harmonic degrees.
        N (int): Maximum FFTLog points.
        x (ndarray): Evaluation grid for the function.
        f (ndarray): Function values to be integrated.
        a (ndarray): Multiplier for Bessel argument.
        b (float): Bias parameter for FFTLog.
        g_r (ndarray): Integration matrix.
        xmin (float): Lower bound of `x`.
        xmax (float): Upper bound of `x`.
        N_interp (int): Interpolation resolution for `a`.
        wind (float): Width of smoothing function (default 0.2).
        fft (FFTLog): FFTLog instance used for coefficient computation.
        power_fft (ndarray): FFTLog exponents.
        step (int): Step size between thick samples in `a`.
        a_thick (ndarray): Thinned `a` sampling used for interpolation.

    Methods:
        int_jl():
            Compute the integral ∫ f(x) * j_l(ax) dx.
        int_k():
            Compute the integral ∫ f(x) * j_l(ax) * k^2 dx.
        int_ddjl():
            Compute the integral with a second-derivative weighting of the Bessel function.
    """

    def __init__(self, **kwargs):
        self.l = kwargs["l"]
        self.N = kwargs["N"]
        self.x = kwargs["x"]
        self.f = kwargs["f"]
        self.a = kwargs["a"]
        self.b = kwargs["b"]
        self.g_r = kwargs["g_r"]
        self.xmin = kwargs["xmin"]
        self.xmax = kwargs["xmax"]
        self.N_interp = kwargs["N_interp"]
        self.wind = 0.2

        fftsettings = dict(
            Nmax=self.N, xmin=self.xmin, xmax=self.xmax, bias=self.b, window=self.wind
        )
        self.fft = FFTLog(**fftsettings)
        self.power_fft = self.fft.Pow

        self.step = int(len(self.a) / self.N_interp)
        self.a_thick = self.a[:: self.step]

    def int_jl(self):
        """
        Compute the integral of the form:

            ∫ f(x) * j_l(ax) dx

        using FFTLog coefficients and interpolation.

        Returns:
            ndarray: Resulting 2D array of shape (len(l), len(a)), the integral values.
        """

        f_thick = jnp.swapaxes(self.f[:, :: self.step], 0, 1)
        w_thick = self.fft.Coef(xin=self.x, f=f_thick, extrap="padding")  # k_thick, N
        w = interp1d(self.a, self.a_thick, w_thick, extrap=True)  # k, N

        k_pow = jax.vmap(lambda p: self.a ** (-1 - p) * 2 ** (-1 + p), 0, 0)(
            self.power_fft
        )
        return jnp.real(
            jnp.sqrt(jnp.pi) * jnp.einsum("lp, kp, pk -> lk", self.g_r, w, k_pow)
        )

    def int_k(self):
        """
        Compute the integral of the form:

            ∫ f(x) * j_l(ax) * a^-2 dx

        using FFTLog coefficients and interpolation.

        Returns:
            ndarray: Resulting 2D array of shape (len(l), len(a)), the integral values.
        """

        f_thick = jnp.swapaxes(self.f[:, :: self.step], 0, 1)
        w_thick = self.fft.Coef(xin=self.x, f=f_thick, extrap="padding")  # k_thick, N
        w = interp1d(self.a, self.a_thick, w_thick, extrap=True)  # k, N

        k_pow = jax.vmap(lambda p: self.a ** (-1 - p) * 2 ** (-1 + p), 0, 0)(
            self.power_fft - 2
        )
        return jnp.real(
            jnp.sqrt(jnp.pi) * jnp.einsum("lp, kp, pk -> lk", self.g_r, w, k_pow)
        )

    def int_ddjl(self):
        """
        Compute the integral with a second-derivative:

            ∫ f(x) * d²/dx²[j_l(ax)] dx

        using FFTLog coefficients and interpolation.

        Returns:
            ndarray: Resulting 2D array of shape (len(l), len(a)), the integral values.
        """

        fftsettings = dict(
            Nmax=self.N,
            xmin=self.xmin,
            xmax=self.xmax,
            bias=self.b + 0.91,
            window=self.wind,
        )
        self.fft = FFTLog(**fftsettings)
        self.power_fft = self.fft.Pow

        f_thick = jnp.swapaxes(self.f[:, :: self.step], 0, 1)
        w_thick = self.fft.Coef(xin=self.x, f=f_thick, extrap="padding")  # k_thick, N
        w = interp1d(self.a, self.a_thick, w_thick, extrap=True)  # k, N

        k_pow = jax.vmap(
            lambda p: self.a ** (-1 - p) * 2 ** (-3 + p) * (-1 + p) * p, 0, 0
        )(self.power_fft)
        return jnp.real(
            jnp.sqrt(jnp.pi) * jnp.einsum("lp, kp, pk -> lk", self.g_r, w, k_pow)
        )
