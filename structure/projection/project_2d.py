# coding:utf-8
import os
import numpy as np
from cosmosis.datablock import names, option_section, BlockError
import re
import sys
import scipy.interpolate as interp
from projection_tools import exact_integral, limber_integral, get_dlogchi, \
                             TomoNzKernel, get_Pk_basis_funcs, get_bias_params_bin, \
                             get_PXX, get_PXm, Enum

# for timing
from timeit import default_timer as timer
from datetime import timedelta

def compute_c1_baseline():
    C1_M_sun = 5e-14  # h^-2 M_S^-1 Mpc^3
    M_sun = 1.9891e30  # kg
    Mpc_in_m = 3.0857e22  # meters
    C1_SI = C1_M_sun / M_sun * (Mpc_in_m)**3  # h^-2 kg^-1 m^3
    # rho_crit_0 = 3 H^2 / 8 pi G
    G = 6.67384e-11  # m^3 kg^-1 s^-2
    H = 100 # h km s^-1 Mpc^-1
    H_SI = H * 1000.0 / Mpc_in_m  # h s^-1
    rho_crit_0 = 3 * H_SI**2 / (8 * np.pi * G)  # h^2 kg m^-3
    f = C1_SI * rho_crit_0
    return f

C1_RHOCRIT = compute_c1_baseline()

class Power3D(object):
    """
    Class representing the 3D power spectrum that enters the Limber calculation.
    Most Power spectra are source-specific, like intrinsic alignments and galaxy
    density.  Others are generic, like the matter power spectrum. The base class
    is not intended to be called directly, and inherited classes must
    specify these:
    - section : str
        name of section to read P(k) from
    - source_specific : bool
        Whether or not this is source-specifc
    and may specify
    - lin_section : str
        A corresponding linear version of the P(k) to use in the exact
        projection calculation.
    """
    def __init__(self, suffix=""):
        """
        Initialize a Power3D instance. This sets the section name
        from which to read in the P(k) data.

        Parameters
        ----------
        suffix: str
            append this suffix to self.section to form section_name
            from which to read in P(k) data.
        """
        self.section_name = self.get_section_name(suffix)
        self.lin_section_name = self.get_section_name(suffix, lin=True)

        # These splines get set later, since they are cosmology dependent
        self.chi_logk_spline = None
        self.lin_z0_logk_spline = None
        self.lin_growth_spline = None
        self.sublin_spline = None

    @classmethod
    def get_section_name(cls, suffix="", lin=False):
        if lin:
            if not hasattr(cls, 'lin_section'):
                name = None
            elif cls.source_specific:
                name = cls.lin_section + suffix
            else:
                name = cls.lin_section
        else:
            if cls.source_specific:
                name = cls.section + suffix
            else:
                name = cls.section
        return name

    def __hash__(self):
        return hash(self.section_name)

    def load_from_block(self, block, chi_of_z):
        """
        Load the z, k, and P(k) values from the block,
        and set the P(chi, log(k)) spline.
        Parameters
        ----------
        block: DataBlock instance
            block from which to read data
        chi_of_z:
            chi(z) spline
        """
        z, k, pk = block.get_grid( self.section_name, "z", "k_h", "p_k" )
        self.z_vals = z
        self.chi_vals = chi_of_z(z)
        self.k_vals = k
        self.logk_vals = np.log(k)
        self.pk_vals = pk

        self.set_chi_logk_spline()

    def set_chi_logk_spline(self):
        """
        Set P(chi, log(k)) spline
        """
        self.chi_logk_spline = interp.RectBivariateSpline(self.chi_vals, self.logk_vals, self.pk_vals)

    def set_nonlimber_splines(self, block, chi_of_z, k_growth=1.e-3, fz_from_block=False):
        """
        Set up various splines etc. needed for the exact projection
        calculation
        For the exact projection integral we need
        i) A P_lin(k,z=0) python spline, this is assigned to self.lin_z0_logk_spline
        ii) A D_lin(chi) python spline, this is assigned to self.lin_growth_spline
        iii) A (P_nl - P_lin) 2d (chi,k) spline to feed into the Limber calculation,
        this is assigned to self.sublin_spline.

        Parameters
        ----------
        block: DataBlock instance
            block from which to read data
        chi_of_z: python spline (e.g. from interp1d or InterpolatedUnivariateSpline)
            chi(z) spline
        k_growth: float
            Value of k (in linear regime) at which to caluclate growth factor from
            the linear power spectrum.
        """
        z_lin, k_lin, P_lin = block.get_grid(self.lin_section_name, "z", "k_h", "p_k")
        chi_lin = chi_of_z(z_lin)

        # Get z=0 slice and set up spline
        P_lin_z0 = P_lin[0,:]
        self.lin_z0_logk_spline = interp.InterpolatedUnivariateSpline(np.log(k_lin), P_lin_z0)

        # Calculate growth for exact integral
        # Use (P(k=k_growth, z)/P(k=k_growth, z=0))^0.5
        # Actually just grab closet slice to k_growth
        growth_ind = (np.abs(k_lin-k_growth)).argmin()
        growth_vals = np.sqrt(np.divide(P_lin[:, growth_ind], P_lin[0, growth_ind],
                    out=np.zeros_like(P_lin[:, growth_ind]), where=P_lin[:, growth_ind]!=0.))
        self.lin_growth_spline = interp.InterpolatedUnivariateSpline(chi_lin, growth_vals)

        # And finally a 2d spline of P_nl - P_lin
        # In the case that P_lin is not exactly separable, we're actually better
        # off using P_nl - D^2(chi)P_lin(0) here since that is what we're using in
        # the exact calculation
        # We may also need to resample the linear P(k) on the nl grid :/
        assert np.allclose(z_lin, self.z_vals)
        P_lin_z0_resamp = interp.interp1d(np.log(k_lin), P_lin_z0,
            kind="cubic", bounds_error=False, fill_value=0.)(self.logk_vals)
        P_lin_from_growth = np.outer(growth_vals**2, P_lin_z0_resamp)
        P_sublin_vals = self.pk_vals - P_lin_from_growth
        self.sublin_spline = interp.RectBivariateSpline(self.chi_vals, self.logk_vals, P_sublin_vals)
        
        if  fz_from_block:
            z_growth = block[names.growth_parameters,'z']
            chi_growth = chi_of_z(z_growth)
            f_growth = block[names.growth_parameters,'f_z']
            self.f_of_chi_spline = interp.InterpolatedUnivariateSpline(chi_growth, f_growth)
        else:             # standard behavior is to do this spline derivative
            # When doing RSD, we also need f(a(\chi)) = dln(D(a(chi)))/dlna
            a_vals = 1/(1+self.z_vals)
            # Make ln(D)(ln(a)) spline
            lnD_of_lna_spline = interp.InterpolatedUnivariateSpline(np.log(a_vals)[::-1], np.log(growth_vals)[::-1])
            # And take derivative
            f = (lnD_of_lna_spline.derivative())(np.log(a_vals))
            # Now can set f(chi) spline.
            self.f_of_chi_spline = interp.InterpolatedUnivariateSpline(self.chi_vals, f)

class MatterPower3D(Power3D):
    section = "matter_power_nl"
    lin_section = "matter_power_lin"
    source_specific = False

class LinearMatterPower3D(Power3D):
    section = "matter_power"
    source_specific = False

class GalaxyPower3D(Power3D):
    section = "galaxy_power"
    lin_section = "galaxy_power_lin"
    source_specific = True

class IntrinsicPower3D(Power3D):
    section = "intrinsic_power"
    source_specific = True

class IntrinsicBBPower3D(Power3D):
    section = "intrinsic_power_bb"
    source_specific = True

class MatterGalaxyPower3D(Power3D):
    section = "matter_galaxy_power"
    source_specific = True

class MatterIntrinsicPower3D(Power3D):
    section = "matter_intrinsic_power"
    source_specific = True

class GalaxyIntrinsicPower3D(Power3D):
    section = "galaxy_intrinsic_power"
    source_specific = True

class WeylPower3D(Power3D):
    section = "weyl_curvature_power_nl"
    source_specific = False

class MatterPower3DPPF(Power3D):
    section = "matter_power_nl"
    lin_section = "matter_power_lin"
    source_specific = False

    def load_from_block(self, block, chi_of_z):
        super().load_from_block(block, chi_of_z)
        # Rescale the P(k) by the D(k, z) function
        z, k, D = block.get_grid("post_friedmann_parameters", "z", "k_h", "D" )
        self.pk_vals *= D

        # Rebuild the spline
        self.set_chi_logk_spline()


def get_lensing_prefactor(block):
    c_kms = 299792.4580
    omega_m = block[names.cosmological_parameters, "omega_m"]
    shear_scaling = 1.5 * (100.0*100.0)/(c_kms*c_kms) * omega_m
    return shear_scaling

def get_lensing_weyl_prefactor(block):
    h = block[names.cosmological_parameters, "h0"]
    shear_weyl_scaling = 1.0 / h**2
    return shear_weyl_scaling

class Spectrum(object):
    """
    Class for calculating a specific type of P(k,z) -> C(l) projection
    integral. Again this base class is not meant to be called directly.
    Child classes should specify:
    autocorrelation: bool
        whether this is an autocorrelation (note e.g. a shear-shear
        correlation between two samples counts as an autocorrelation
        in this context)
    kernel_types: (str, str)
        tuple specifying the two kernel types e.g. "W" for lensing,
        "N" for number counts
    name: str
        default section name for this spectrum e.g. shear_cl
    prefactor_type: (str, str)
        tuple of prefactor type e.g. "lensing", None, "mag"
    """
    autocorrelation = False
    # These should make it more obvious if the values are not overwritten by subclasses
    kernel_types = ("?", "?")
    name = "?"
    prefactor_type = ("", "")

    def __init__(self, source, sample_a, sample_b, power_key, save_name="", only_bins=None, len_only_bins=None):
        # caches of n(z), w(z), P(k,z), etc.
        self.source = source
        self.sample_a, self.sample_b = sample_a, sample_b
        self.power_key = power_key
        self.save_name = save_name
        if self.save_name:
            self.section_name = self.name + "_" + self.save_name
        else:
            self.section_name = self.name
        self.only_bins = only_bins
        self.len_only_bins = len_only_bins
        self.input_section_name = power_key[0].section + power_key[1]

    def should_do_bin(self, i):
        return self.only_bins is None or i in self.only_bins

    def nbins(self):
        na = self.source.kernels[self.sample_a].nbin
        nb = self.source.kernels[self.sample_b].nbin
        return na, nb

    def is_autocorrelation(self):
        """
        This is an autocorrelation if the basic type is an auto-correlation
        (e.g. shear-shear, position-position, but not shear position)
        and the two n(z) samples are the same.
        """
        return self.autocorrelation and (self.sample_a == self.sample_b)

    @classmethod
    def option_name(cls):
        """Convert the CamelCase name to hypen-separated.
        For example ShearShear becomes shear-shear
        """
        name = cls.__name__
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1-\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1-\2', s1).lower()

    def get_prefactor(self, block, bin1, bin2):
        """
        Get prefactors for C(l) (e.g. for lensing or magnification)
        """
        prefactor = 1
        # first prefactor
        if self.prefactor_type[0] is None:
            pass
        elif self.prefactor_type[0]=="lensing":
            prefactor *= self.source.lensing_prefactor
        elif self.prefactor_type[0]=="lensing_weyl":
            prefactor *= self.source.lensing_weyl_prefactor
        elif self.prefactor_type[0]=="mag":
            prefactor *= self.get_magnification_prefactor(block,
                self.sample_a, bin1) * self.source.lensing_prefactor

        # second prefactor
        if self.prefactor_type[1] is None:
            pass
        elif self.prefactor_type[1]=="lensing":
            prefactor *= self.source.lensing_prefactor
        elif self.prefactor_type[1]=="lensing_weyl":
            prefactor *= self.source.lensing_weyl_prefactor
        elif self.prefactor_type[1]=="mag":
            prefactor *= self.get_magnification_prefactor(block,
                self.sample_b, bin2) * self.source.lensing_prefactor

        return prefactor

    # Some spectra include a magnification term for one or both bins.
    # In those cases an additional term which quantifies the response
    # of the number density to kappa is required as a prefactor
    # We use the convention from 0911.2454 such that this prefactor
    # is 2*(alpha-1). For a given sample and redshift bin i, alpha is
    # read from section "mag_alpha_<sample>" and value alpha_<i>
    # e.g. mag_alpha_redmagic, alpha_3 for the 3rd redshift bin of sample
    # redmagic
    def get_magnification_prefactor(self, block, sample, bin_num):
        alpha = block[ f"mag_alpha_{sample}", f"alpha_{bin_num}" ]
        return 2 * (alpha - 1)

    def compute_shared_chi_range(self, sig_over_dchi):
        """
        Compute the minimum and maximum chi for the Limber integral
        over all of the bins, and a suitable dchi spacing.
        """
        chi_min = np.inf
        chi_max = -np.inf
        dchi = np.inf
        kernels = self.source.kernels[self.sample_a]
        for i in range(1, kernels.nbin+1):
            spl = kernels.get_kernel_spline(self.kernel_types[0], i)
            chi_min = min(chi_min, spl.xmin_clipped)
            chi_max = max(chi_max, spl.xmax_clipped)
            dchi = min(dchi, spl.sigma / sig_over_dchi)
        return float(chi_min), float(chi_max), float(dchi)
    def compute_limber(self, block, ell, bin1, bin2, dchi=None, sig_over_dchi=100.,
        chimin=None, chimax=None):
        r"""
        Calculate the Limber integral
        C(l) = \int d(chi) K_a(chi) K_b(chi) P(ell+0.5/chi, chi) / chi^2

        Most common usage will be to set dchi, chimin, chimax = None
        chimin and chimax will then be extracted from the kernels,
        and dchi will be calculated from the kernel width and sig_over_dchi

        Parameters
        ----------
        block: DataBlock instance
            block from which to read data
        ell: float array
            np.array of ell values at which to calculate C(l)
        bin1: int
            first tomographic bin index (starting from 1)
        bin2: int
            second tomographic bin index (starting from 2)
        dchi: float
            spacing in dchi at which to compute integral
        sig_over_dchi: float
            ratio between width of kernel sigma and dchi
        chimin:
            minimum chi for integral over chi
        chimax:
            maximum chi for integral over chi

        Returns
        -------
        c_ell: float array
            array of C(l) values
        """
        # Get the required 2d power spline P(chi, logk)
        P_chi_logk_spline = self.get_power_spline(block, bin1, bin2)
        
        # Get the kernels
        K1 = self.source.kernels[self.sample_a].get_kernel_spline(self.kernel_types[0], bin1)
        K2 = self.source.kernels[self.sample_b].get_kernel_spline(self.kernel_types[1], bin2)

        # Need to choose a chimin, chimax and dchi for the integral.
        # By default we use the clipped range of the integral. The spacing
        # is the 1 sigma width of the n(z) / the sig_over_dchi parameter, which
        # is 100 by default, but can be adjusted.
        if chimin is None:
            chimin = max(K1.xmin_clipped, K2.xmin_clipped)
        if chimax is None:
            chimax = min(K1.xmax_clipped, K2.xmax_clipped)
        if dchi is None:
            assert (sig_over_dchi is not None)
            dchi = min(K1.sigma / sig_over_dchi, K2.sigma / sig_over_dchi)

        # Run the main Limber integral
        c_ell, c_ell_err = limber_integral(ell, K1, K2, P_chi_logk_spline, chimin,
            chimax, dchi, interpolation_cache=self.source.interpolation_cache)

        # Rescale by the h, Omega_m, etc factor, which depends which spectrum
        # us being computed
        c_ell *= self.get_prefactor(block, bin1, bin2)

        return c_ell

    def compute_exact(self, block, ell, bin1, bin2, dlogchi=None,
        sig_over_dchi=20.0, chi_pad_lower=2.0, chi_pad_upper=2.0,
        chimin=None, chimax=None, dchi_limber=None, do_rsd=False):
        """
        The 'exact' calculation is in two parts:
        Non-limber for the separable linear contribution, and
        Limber for the non-linear part (P_nl - P_lin).

        Parameters
        ----------
        block: DataBlock instance
            block from which to read data
        ell: float array
            np.array of ell values at which to calculate C(l)
        bin1: int
            first tomographic bin index (starting from 1)
        bin2: int
            second tomographic bin index (starting from 2)
        dchi: float
            spacing in dchi at which to compute integral
        sig_over_dchi: float
            ratio between width of kernel sigma and dchi
        chimin:
            minimum chi for integral over chi
        chimax:
            maximum chi for integral over chi
        chi_pad_lower:
            extend the integral over log(chi) lower limit by
            this factor (may be required for good fftlog behaviour)
        chi_pad_upper:
            extend the integral over log(chi) upper limit by
            this factor (may be required for good fftlog behaviour)
        dchi_limber:
            chi spacing to use for Limber integral part of calculation

        Returns
        -------
        c_ell: float array
            array of C(l) values
        """

        # This base class doesn't handle Redshift Space Distortions,
        # so raise error if do_rsd=True. The Lingal-Lingal subclass is
        # the only one that does.
        if do_rsd:
            raise ValueError(f"{self.__name__} can't handle do_rsd=True")

        # Get the kernel splines
        kernels = self.source.kernels
        K1 = kernels[self.sample_a].get_kernel_spline(self.kernel_types[0], bin1)
        K2 = kernels[self.sample_b].get_kernel_spline(self.kernel_types[1], bin2)

        # Get the P(log(k), z=0) and growth(chi) splines
        lin_z0_logk_spline, lin_growth_spline = self.get_lin_power_growth(block, bin1, bin2)

        # Need to choose a chimin, chimax and dchi for the integral.
        # By default
        if chimin is None:
            chimin = max( K1.xmin_clipped, K2.xmin_clipped )
        if chimax is None:
            chimax = min( K1.xmax_clipped, K2.xmax_clipped )
        if dlogchi is None:
            sig = min( K1.sigma/sig_over_dchi, K2.sigma/sig_over_dchi )
            # We want a maximum dchi that is sig / sig_over_dchi where sig is the
            # width of the narrower kernel.
            dchi = sig / sig_over_dchi
            dlogchi = get_dlogchi(dchi, chimax)

        # Call the exact calculation with linear P(k) spline
        c_ell = exact_integral(ell, K1, K2, lin_z0_logk_spline, lin_growth_spline,
            chimin, chimax, dlogchi, chi_pad_upper=chi_pad_upper,
            chi_pad_lower=chi_pad_lower)

        # Now call Limber integral with P_nl-P_lin
        P_sublin_spline = self.get_power_sublin(block, bin1, bin2)

        # We can either specify the chi spacing manually or the sig_over_chi
        if dchi_limber is None:
            assert (sig_over_dchi is not None)
            dchi_limber = min( K1.sigma/sig_over_dchi, K2.sigma/sig_over_dchi )

        c_ell_sublin,_ = limber_integral(ell, K1, K2, P_sublin_spline,
            chimin, chimax, dchi_limber, interpolation_cache=self.source.interpolation_cache)

        # Add the Limber on the non-linear part to the total
        c_ell += c_ell_sublin

        # Scale by the (again, spectrum-specific) prefactor.
        c_ell *= self.get_prefactor(block, bin1, bin2)

        return c_ell

    def compute(self, block, ell_limber, bin1, bin2,
        dchi_limber=None, sig_over_dchi_limber=100., chimin=None, chimax=None,
        ell_exact=None, exact_kwargs=None, cut_ell_limber=True):
        """
        Do the C(l) calculation via calling compute_limber and/or compute_exact.

        The Limber calculation is done for the complete range, and then
        if the ell_exact parameter is given an as array of ell values for
        which the non-limber calculation is needed then these are concatenated
        with the limber C_ell from everything with higher ell values than that.

        Parameters
        ----------
        block: DataBlock instance
            block from which to read data
        ell_limber: float array
            np.array of ell values at which to calculate C(l) via Limber
        bin1: int
            first tomographic bin index (starting from 1)
        bin2: int
            second tomographic bin index (starting from 2)
        dchi_limber: float
            spacing in dchi at which to compute limber integral
        sig_over_dchi_limber: float
            ratio between width of kernel sigma and dchi for limber integral
        chimin: float
            minimum chi for integral over chi
        chimax: float
            maximum chi for integral over chi
        ell_exact: np.array
            ell values for exact integral
        exact_kwargs: dict
            Dictionary of keyword arguments for the exact integral
        """

        if ell_exact is not None:
            if cut_ell_limber:
                ell_limber = ell_limber[ell_limber>=ell_exact[-1]]

        # Do the Limber integral
        cl_limber = self.compute_limber(block, ell_limber, bin1, bin2,
            dchi=dchi_limber, sig_over_dchi=sig_over_dchi_limber,
            chimin=chimin, chimax=chimax)

        ell_out = ell_limber
        cl_out = cl_limber

        if ell_exact is not None:
            # Perform the exact calculation for the requested ell values
            cl_exact = self.compute_exact(block, ell_exact, bin1, bin2,
                                          **exact_kwargs)

            # Combine Limber and non-Limber into one output array
            use_limber = ell_limber > ell_exact[-1]
            ell_out = np.concatenate((ell_exact, ell_limber[use_limber]))
            cl_out = np.concatenate((cl_exact, cl_limber[use_limber]))

        return ell_out, cl_out

    def get_power_spline(self, block, bin1, bin2):
        return self.pk_chi_logk_spline

    def get_power_sublin(self, block, bin1, bin2):
        return self.pk_sublin_spline

    def get_lin_power_growth(self, block, bin1, bin2):
        return self.pk_lin_z0_logk_spline, self.lin_growth_spline

    def prepare(self, block, **kwargs):
        """This is called prior to calculating the C(l)s
        for a given spectrum and sets the P(k) splines
        required for the calculation. For the base
        class this stuff could be done in __init__
        but for child classes having this extra method
        is more useful.
        """
        p = self.source.power[self.power_key]
        self.pk_chi_logk_spline = p.chi_logk_spline
        self.pk_sublin_spline = p.sublin_spline
        self.pk_lin_z0_logk_spline = p.lin_z0_logk_spline
        self.lin_growth_spline = p.lin_growth_spline
        return 0

class LingalLingalSpectrum(Spectrum):
    """
    Class for calculating clustering C(l) for a linearly biased galaxy
    sample. We overwrite the exact calculation to include RSD.
    """
    def compute_exact(self, block, ell, bin1, bin2, dlogchi=None,
     sig_over_dchi=10., chi_pad_lower=2., chi_pad_upper=2.,
     chimin=None, chimax=None, dchi_limber=None, do_rsd=True):
        # Same as Spectrum.compute_exact, with the do_rsd option
        # allowed.

        # Get the kernels
        K1 = (self.source.kernels[self.sample_a]).get_kernel_spline(self.kernel_types[0], bin1)
        K2 = (self.source.kernels[self.sample_b]).get_kernel_spline(self.kernel_types[1], bin2)

        lin_z0_logk_spline, lin_growth_spline  = self.get_lin_power_growth(block,
                                                                           bin1,
                                                                           bin2)

        # Need to choose a chimin, chimax and dchi for the integral.
        # By default following the same rules as the parent class
        if chimin is None:
            chimin = max( K1.xmin_clipped, K2.xmin_clipped )
        if chimax is None:
            chimax = min( K1.xmax_clipped, K2.xmax_clipped )
        if dlogchi is None:
            sig = min(K1.sigma / sig_over_dchi, K2.sigma / sig_over_dchi)
            dchi = sig / sig_over_dchi
            dlogchi = get_dlogchi(dchi, chimax)

        # Exact calculation with linear P(k)
        if do_rsd:
            # Get f(chi) growth spline
            f_of_chi_spline = (self.source.power[self.power_key]).f_of_chi_spline

            # and call integral with do_rsd=True and passing linear bias
            # and f(chi)
            c_ell = exact_integral(ell, K1, K2, lin_z0_logk_spline, lin_growth_spline,
            chimin, chimax, dlogchi, do_rsd=True, b1_1=self.lin_bias_values_a[bin1],
            b1_2=self.lin_bias_values_b[bin2], f_interp=f_of_chi_spline,
            chi_pad_upper=chi_pad_upper, chi_pad_lower=chi_pad_lower,
                )
        else:
            c_ell = exact_integral(ell, K1, K2, lin_z0_logk_spline, lin_growth_spline,
                 chimin, chimax, dlogchi, do_rsd=False, b1_1=self.lin_bias_values_a[bin1],
                 b1_2=self.lin_bias_values_b[bin2], chi_pad_upper=chi_pad_upper,
                chi_pad_lower=chi_pad_lower)

        # Limber with P_nl - P_lin
        P_sublin_spline = self.get_power_sublin(block, bin1, bin2)

        if dchi_limber is None:
            assert (sig_over_dchi is not None)
            dchi_limber = min( K1.sigma/sig_over_dchi, K2.sigma/sig_over_dchi )

        c_ell_sublin,_ = limber_integral(ell, K1, K2, P_sublin_spline, chimin,
            chimax, dchi_limber, interpolation_cache=self.source.interpolation_cache)

        # We apply the linear bias here to the NL-only (sublin) part.  The
        # linear bias is already applied in the exact_integral above to the
        # linear part
        c_ell_sublin *= self.lin_bias_values_a[bin1] * self.lin_bias_values_b[bin2]

        # Combine, rescale, and return the total spectrum
        c_ell += c_ell_sublin
        c_ell *= self.get_prefactor(block, bin1, bin2)
        return c_ell

    def compute_limber(self, block, ell, bin1, bin2, dchi=None, sig_over_dchi=100.,
        chimin=None, chimax=None):

        # same as the base class but we multiply by galaxy bias
        c_ell = super().compute_limber(block,
            ell, bin1, bin2, dchi=dchi, sig_over_dchi=sig_over_dchi,
            chimin=chimin, chimax=chimax)
        return c_ell * self.lin_bias_values_a[bin1] * self.lin_bias_values_b[bin2]

    def prepare(self, block, lin_bias_prefix="b", **kwargs):
        # Call the parent prepare
        super().prepare(block)

        # We also need the linear bias values for each bin
        self.lin_bias_values_a = {}
        self.lin_bias_values_b = {}

        # Make sure kernel_types are both "N", indicating number density.
        # There will not be linear bias values for the sample otherwise,
        # and this whole class is designed for linearly-biased densities.
        assert self.kernel_types[0] == self.kernel_types[1] == "N"
        nbin = self.source.kernels[self.sample_a].nbin

        for i in range(1, nbin+1):
            sec = f"bias_{self.sample_a}"
            val = f"{lin_bias_prefix}{i}"
            self.lin_bias_values_a[i] = block[sec, val]

        # If the samples are the same (e.g. lens-lens)
        # then the set of bias valus are the same for each sample too.
        # Otherwise (i.e. lens1-lens2) then load the second suite
        if self.sample_b == self.sample_a:
            self.lin_bias_values_b = self.lin_bias_values_a
        else:
            nbin = self.source.kernels[self.sample_b].nbin
            for i in range(1, nbin+1):
                sec = f"bias_{self.sample_b}"
                val = f"{lin_bias_prefix}{i}"
                self.lin_bias_values_b[i] = block[sec, val]
        return 0

class LingalLensingSpectrum(Spectrum):
    """
    A spectrum for the cross-correlation between a linearly-biased
    density sample and a shear sample, i.e. galaxy-galaxy lensing
    with linear bias.
    """

    def compute_exact(self, block, ell, bin1, bin2, dlogchi=None,
        sig_over_dchi=10., chi_pad_lower=2., chi_pad_upper=2.,
        chimin=None, chimax=None, dchi_limber=None, do_rsd=False):

        if do_rsd:
            raise NotImplementedError("RSD not yet implemented for LingalLensingSpectrum")

        # Call the exact calculation from the base class
        c_ell = super().compute_exact(block,
            ell, bin1, bin2, dlogchi=dlogchi, sig_over_dchi=sig_over_dchi,
            chi_pad_lower=chi_pad_lower, chi_pad_upper=chi_pad_upper,
            chimin=chimin, chimax=chimax, dchi_limber=dchi_limber,
            do_rsd=do_rsd)

        # Apply a linear bias and return
        c_ell *= self.lin_bias_values_a[bin1]
        return c_ell

    def compute_limber(self, block, ell, bin1, bin2, dchi=None, sig_over_dchi=100.,
        chimin=None, chimax=None):

        # The limber spectrum here is identical to the parent class value
        # except we have to apply the linear bias.
        c_ell = super().compute_limber(block,
            ell, bin1, bin2, dchi=dchi, sig_over_dchi=sig_over_dchi,
            chimin=chimin, chimax=chimax)
        return c_ell * self.lin_bias_values_a[bin1]

    def prepare(self, block, lin_bias_prefix="b", **kwargs):
        # Call the parent prepare, which pulls the power splines out
        # of the source (which has read them from the data block)
        super().prepare(block)

        # Then also load in bias values for first sample.
        # Unless things have gone very wrong this should be
        # a number density spectrum
        assert self.kernel_types[0] == "N"

        # We only need the bias values for the "A" sample,
        # since the B sample is a lensing sample.
        self.lin_bias_values_a = {}
        nbin = self.source.kernels[self.sample_a].nbin
        for i in range(1, nbin+1):
            sec = f"bias_{self.sample_a}"
            val = f"{lin_bias_prefix}{i}"  # usually b1, b2, b3, ...
            self.lin_bias_values_a[i] = block[sec, val]


class NlgalNlgalSpectrum(LingalLingalSpectrum):
    """
    Class for computing galaxy C(l) with non-linear bias models.

    For this case the linear power spectrum used in the exact
    calculation is still just the linear matter power spectrum
    (and the b_1 values are passed along with this).

    We also construct the non-linear P(k) using fast-pt, on a bin
    pair-by-bin pair basis. This non-linear P(k) is used for the
    Limber calculation, and to construct the P_nl-P_lin spline
    used for the non-linear correction in the exact calculation.

    Since we inherit the compute method from LingalLingal, which
    applies the galaxy bias factors after the computing the C(l)s
    we divide P_nl by the linear bias factors when setting
    self.pk_chi_logk_spline and self.pk_sublin_spline.

    We might want to revise how that all works at some point
    as it seems a bit clumsy. But oh well.
    """
    def prepare(self, block, pt_type="oneloop_eul_bk", **kwargs):

        self.pt_type = pt_type

        # Load bias values into a dictionary
        self.bias_values_a, self.lin_bias_values_a = self.load_bias(
            block, self.sample_a, pt_type)
        if self.sample_a == self.sample_b:
            self.bias_values_b = self.bias_values_a
            self.lin_bias_values_b = self.lin_bias_values_a
        else:
            self.bias_values_b, self.lin_bias_values_b = self.load_bias(
                block, self.sample_b, pt_type)

        # Get power spectrum terms from fastpt. These
        # will be used later to form P_gg for a given bin
        # combination
        self.k_nl_bias, self.Pk_basis_funcs = get_Pk_basis_funcs(
            block, self.pt_type, output_nl_grid=True)

        p = self.source.power[self.power_key]
        self.pk_chi_logk_spline = None # Set these to None as
        self.pk_sublin_spline = None # they depend on the bin pair
        self.pk_lin_z0_logk_spline = p.lin_z0_logk_spline
        self.lin_growth_spline = p.lin_growth_spline

        # get_power_spline and get_power_sublin check these
        # attributes to see whether they need to call set_power
        self.current_bin1, self.current_bin2 = None, None

    def load_bias(self, block, sample, pt_type):
        # Load bias values from the block
        nbin = self.source.kernels[sample].nbin
        bias_values = {}
        lin_bias_values = {}
        for bin_num in range(1, nbin+1):
            bias_values[bin_num] = get_bias_params_bin(block,
                bin_num, pt_type, f"bias_{sample}")
            lin_bias_values[bin_num] = bias_values[bin_num]["b1E"]
        return bias_values, lin_bias_values

    # The P(k)s for the non-linear bias classes work
    # a bit differently, annoyingly.
    def set_power(self, block, bin1, bin2):
        # Record the current bin pair
        self.current_bin1, self.current_bin2 = bin1, bin2

        # Get the full non-linear P_gg(k) from fastpt
        P_gg, P_gg_terms = get_PXX(self.bias_values_a[bin1],
            self.bias_values_b[bin2], self.Pk_basis_funcs,
            self.pt_type)

        # We inherit the compute method (and thus
        # compute_limber and compute_exact methods)
        # from LingalLingalSpectrum, which expects the
        # non-linear P(k) in self.pk_chi_logk_spline, and
        # then multiplies by the galaxy bias after the fact.
        # So we set self.pk_chi_logk_spline to
        # P_gg / lin_bias_1 / lin_bias_2
        blin_1 = self.lin_bias_values_a[bin1]
        blin_2 = self.lin_bias_values_b[bin2]
        P_gg_div_bias = (P_gg / blin_1 / blin_2)
        chi_vals = self.source.power[self.power_key].chi_vals
        self.pk_chi_logk_spline = interp.RectBivariateSpline(
            chi_vals, np.log(self.k_nl_bias),
            P_gg_div_bias)
        # Now we need the P-P_lin term to use in the
        # Limber part of compute_exact. Again, this should
        # be divided by the linear bias factors, since
        # these are applied in LingalLingalSpectrum.compute_exact
        P_sublin = (P_gg_div_bias -
            self.Pk_basis_funcs["Plin_from_growth"])
        self.pk_sublin_spline = interp.RectBivariateSpline(
            chi_vals, np.log(self.k_nl_bias),
            P_sublin)

    def get_power_spline(self, block, bin1, bin2):
        # If current bin pair does not match bin1, bin2,
        # call self.set_power
        if self.current_bin1 != bin1 or self.current_bin2 != bin2:
            self.set_power(block, bin1, bin2)
        return self.pk_chi_logk_spline

    def get_power_sublin(self, block, bin1, bin2):
        # If current bin pair does not match bin1, bin2,
        # call self.set_power
        if (self.current_bin1 != bin1 or self.current_bin2 != bin2):
            self.set_power(block, bin1, bin2)
        return self.pk_sublin_spline

class NlgalShearSpectrum(LingalLensingSpectrum):
    """
    Class for computing galaxy C(l) with non-linear bias models. For this case
    the linear power spectrum used in the exact calculation is still
    just the linear matter power spectrum (and the b_1 values are passed
    along with this). We also need to construct the non-Linear P(k) using fast-pt.
    """
    def prepare(self, block, pt_type="oneloop_eul_bk", **kwargs):
        # assign pt_type
        self.pt_type = pt_type

        # Load bias values into a dictionary
        self.bias_values_a, self.lin_bias_values_a = self.load_bias(
            block, self.sample_a, pt_type)

        # Get power spectrum terms from fastpt. These
        # will be used later to form P_gm for a given bin
        # combination
        self.k_nl_bias, self.Pk_basis_funcs = get_Pk_basis_funcs(
            block, self.pt_type, output_nl_grid=True)

        p = self.source.power[self.power_key]
        self.pk_chi_logk_spline = None # Set these to None as
        self.pk_sublin_spline = None # they depend on the bin pair
        self.pk_lin_z0_logk_spline = p.lin_z0_logk_spline
        self.lin_growth_spline = p.lin_growth_spline

        # get_power_spline and get_power_sublin check these
        # attributes to see whether they need to call set_power
        self.current_bin1 = None

    def load_bias(self, block, sample, pt_type):
        # Load bias values from the block
        nbin = self.source.kernels[sample].nbin
        bias_values = {}
        lin_bias_values = {}
        for bin_num in range(1, nbin+1):
            bias_values[bin_num] = get_bias_params_bin(block,
                bin_num, pt_type, f"bias_{sample}")
            lin_bias_values[bin_num] = bias_values[bin_num]["b1E"]
        return bias_values, lin_bias_values

    # The P(k)s for the non-linear bias classes work
    # a bit differently, annoyingly.
    def set_power(self, block, bin1, bin2):
        # Record the current bin pair
        self.current_bin1 = bin1

        # Get the full non-linear P_gg(k) from fastpt
        P_gm, P_gm_terms = get_PXm(self.bias_values_a[bin1],
            self.Pk_basis_funcs, self.pt_type)

        # We inherit the compute method (and thus
        # compute_limber and compute_exact methods)
        # from LingalLensingSpectrum, which expects the
        # non-linear P(k) in self.pk_chi_logk_spline, and
        # then multiplies by the galaxy bias after the fact.
        # So we set self.pk_chi_logk_spline to
        # P_gm / lin_bias_1
        blin_1 = self.lin_bias_values_a[bin1]
        P_gm_div_bias = (P_gm / blin_1 )
        chi_vals = self.source.power[self.power_key].chi_vals
        self.pk_chi_logk_spline = interp.RectBivariateSpline(
            chi_vals, np.log(self.k_nl_bias),
            P_gm_div_bias)

        # Now we need the P-P_lin term to use in the
        # Limber part of compute_exact. Again, this should
        # be divided by the linear bias factors, since
        # $hese are applied in LingalLingalSpectrum.compute_exact
        P_sublin = (P_gm_div_bias - self.Pk_basis_funcs["Plin_from_growth"])
        self.pk_sublin_spline = interp.RectBivariateSpline(
            chi_vals, np.log(self.k_nl_bias),
            P_sublin)

    def get_power_spline(self, block, bin1, bin2):
        # If current bin pair does not match bin1, bin2,
        # call self.set_power
        if self.current_bin1!=bin1:
            self.set_power(block, bin1, bin2)
        return self.pk_chi_logk_spline

    def get_power_sublin(self, block, bin1, bin2):
        # If current bin pair does not match bin1, bin2,
        # call self.set_power
        if self.current_bin1!=bin1:
            self.set_power(block, bin1, bin2)
        return self.pk_sublin_spline


# This is pretty cool.
# You can make an enumeration class which
# contains a list of possible options for something.
# But the options can each be anything - full python objects, and in this
# case they are classes of spectrum. So we can easily look these up by name,
# loop through them, etc.
class SpectrumType(Enum):
    class ShearShear(Spectrum):
        power_3d_type = MatterPower3D
        kernel_types = ("W", "W")
        autocorrelation = True
        name = names.shear_cl
        prefactor_type = ("lensing", "lensing")
        has_rsd = False

    class WeylWeyl(Spectrum):
        power_3d_type = WeylPower3D
        kernel_types = ("W_W", "W_W")
        autocorrelation = True
        name = "shear_cl"
        prefactor_type = ("lensing_weyl","lensing_weyl")
        has_rsd = False

    class PpfshearPpfshear(Spectrum):
        power_3d_type = MatterPower3DPPF
        kernel_types = ("W", "W")
        autocorrelation = True
        name = names.shear_cl
        prefactor_type = ("lensing", "lensing")
        has_rsd = False


    class ShearIntrinsic(Spectrum):
        power_3d_type = MatterIntrinsicPower3D
        kernel_types = ("W", "N")
        autocorrelation = False
        name = names.shear_cl_gi
        prefactor_type = ("lensing", None)
        has_rsd = False

    class IntrinsicIntrinsic(Spectrum):
        power_3d_type = IntrinsicPower3D
        kernel_types = ("N", "N")
        autocorrelation = True
        name = names.shear_cl_ii
        prefactor_type = (None, None)
        has_rsd = False

    class IntrinsicbIntrinsicb(Spectrum):
        power_3d_type = IntrinsicBBPower3D
        kernel_types = ("N", "N")
        autocorrelation = True
        name = "shear_cl_bb"
        prefactor_type = (None, None)
        has_rsd = False

    class PositionPosition(Spectrum):
        power_3d_type = MatterPower3D
        kernel_types = ("N", "N")
        autocorrelation = True
        name = "galaxy_cl"
        prefactor_type = (None, None)
        has_rsd = False

    class MagnificationDensity(Spectrum):
        power_3d_type = MatterPower3D
        kernel_types = ("W", "N")
        autocorrelation = False
        name = "magnification_density_cl"
        prefactor_type = ("mag", None)
        has_rsd = False

    class MagnificationMagnification(Spectrum):
        power_3d_type = MatterPower3D
        kernel_types = ("W", "W")
        autocorrelation = True
        name = "magnification_cl"
        prefactor_type = ("mag", "mag")
        has_rsd = False

    class PositionShear(Spectrum):
        power_3d_type = MatterPower3D
        kernel_types = ("N", "W")
        autocorrelation = False
        name = "galaxy_shear_cl"
        prefactor_type = (None, "lensing")
        has_rsd = False

    class DensityIntrinsic(Spectrum):
        power_3d_type = MatterIntrinsicPower3D
        kernel_types = ("N", "N")
        autocorrelation = False
        name = "galaxy_intrinsic_cl"
        prefactor_type = (None, None)
        has_rsd = False

    class MagnificationIntrinsic(Spectrum):
        power_3d_type = MatterIntrinsicPower3D
        kernel_types = ("W", "N")
        autocorrelation = False
        name = "magnification_intrinsic_cl"
        prefactor_type = ("mag", None)
        has_rsd = False

    class MagnificationShear(Spectrum):
        power_3d_type = MatterPower3D
        kernel_types = ("W", "W")
        autocorrelation = False
        name = "magnification_shear_cl"
        prefactor_type = ("mag", "lensing")
        has_rsd = False

    class ShearCmbkappa(Spectrum):
        power_3d_type = MatterPower3D
        kernel_types = ("W", "K")
        autocorrelation = False
        name = "shear_cmbkappa_cl"
        prefactor_type = ("lensing", "lensing")
        has_rsd = False

    class CmbkappaCmbkappa(Spectrum):
        power_3d_type = MatterPower3D
        kernel_types = ("K", "K")
        autocorrelation = True
        name = "cmbkappa_cl"
        prefactor_type = ("lensing", "lensing")
        has_rsd = False

    class IntrinsicCmbkappa(Spectrum):
        power_3d_type = MatterIntrinsicPower3D
        kernel_types = ("N", "K")
        autocorrelation = False
        name = "intrinsic_cmbkappa_cl"
        prefactor_type = (None, "lensing")
        has_rsd = False

    class LingalCmbkappa(LingalLensingSpectrum):
        power_3d_type = MatterPower3D
        kernel_types = ("N", "K")
        autocorrelation = False
        name = "galaxy_cmbkappa_cl"
        prefactor_type = (None, "lensing")
        has_rsd = False

    class MagnificationCmbkappa(Spectrum):
        power_3d_type = MatterPower3D
        kernel_types = ("W", "K")
        autocorrelation = False
        name = "magnification_cmbkappa_cl"
        prefactor_type = ("mag", "lensing")
        has_rsd = False

    class DensityCmbkappa(Spectrum):
        power_3d_type = MatterPower3D
        kernel_types = ("N", "K")
        autocorrelation = False
        name = "galaxy_cmbkappa_cl"
        prefactor_type = (None, "lensing")
        has_rsd = False

    class FastShearShearIA(Spectrum):
        """
        Variant method of Shear+IA calculation that
        does the integral all at once including the shear
        components.  Only works for IA with P_II(k)
        and P_GI(k) proportional to the
        matter power spectrum.
        """
        power_3d_type = MatterPower3D
        kernel_types = ("F", "F")
        autocorrelation = True
        name = names.shear_cl
        prefactor_type = ("lensing","lensing")
        has_rsd = False

    class FastLingalShearIA(LingalLensingSpectrum):
        autocorrelation = False
        power_3d_type = MatterPower3D
        kernel_types = ("N", "F")
        name = "galaxy_shear_cl"
        prefactor_type = (None, "lensing")
        has_rsd = False

    class FastPositionShearIA(Spectrum):
        power_3d_type = MatterPower3D
        kernel_types = ("N", "F")
        autocorrelation = False
        name = "galaxy_shear_cl"
        prefactor_type = (None, "lensing")
        has_rsd = False

    class LingalLingal(LingalLingalSpectrum):
        power_3d_type = MatterPower3D
        kernel_types = ("N", "N")
        autocorrelation = True
        name = "galaxy_cl"
        prefactor_type = (None, None)
        has_rsd = True

    class LingalShear(LingalLensingSpectrum):
        power_3d_type = MatterPower3D
        kernel_types = ("N", "W")
        autocorrelation = False
        name = "galaxy_shear_cl"
        prefactor_type = (None, "lensing")
        has_rsd = False

    class LingalMagnification(LingalLensingSpectrum):
        autocorrelation = False
        power_3d_type = MatterPower3D
        kernel_types = ("N", "W")
        autocorrelation = False
        name = "galaxy_magnification_cl"
        prefactor_type = (None, "mag")
        has_rsd = False

    class LingalIntrinsic(LingalLensingSpectrum):
        power_3d_type = MatterIntrinsicPower3D
        kernel_types = ("N", "N")
        autocorrelation = False
        name = "galaxy_intrinsic_cl"
        prefactor_type = (None, None)
        has_rsd = False

    class NlgalNlgal(NlgalNlgalSpectrum):
        power_3d_type = MatterPower3D
        kernel_types = ("N", "N")
        autocorrelation = True
        name = "galaxy_cl"
        prefactor_type = (None, None)
        has_rsd = True

    class NlgalShear(NlgalShearSpectrum):
        autocorrelation = False
        power_3d_type = MatterPower3D
        kernel_types = ("N", "W")
        autocorrelation = False
        name = "galaxy_shear_cl"
        prefactor_type = (None, "lensing")
        has_rsd = False

    class NlgalMagnification(NlgalShearSpectrum):
        autocorrelation = False
        power_3d_type = MatterPower3D
        kernel_types = ("N", "W")
        autocorrelation = False
        name = "galaxy_magnification_cl"
        prefactor_type = (None, "mag")
        has_rsd = False

    class GenericclusteringGenericclustering(Spectrum):
        autocorrelation = True
        power_3d_type = GalaxyPower3D
        kernel_types = ("N", "N")
        name = "galaxy_cl"
        prefactor_type = (None, None)
        has_rsd = True

    class GenericclusteringShear(Spectrum):
        power_3d_type = MatterGalaxyPower3D
        kernel_types = ("N", "W")
        autocorrelation = False
        name = "galaxy_shear_cl"
        prefactor_type = (None, "lensing")
        has_rsd = False

    class GenericclusteringIntrinsic(Spectrum):
        power_3d_type = GalaxyIntrinsicPower3D
        kernel_types = ("N", "N")
        autocorrelation = False
        name = "galaxy_intrinsic_cl"
        prefactor_type = (None, None)
        has_rsd = False


class SpectrumCalculator(object):
    # It is useful to put this here so we can subclass to add new spectrum
    # types, for example ones done with modified gravity changes.
    spectrumType = SpectrumType

    def __init__(self, options):
        # General options
        self.verbose = options.get_bool(option_section, "verbose", False)
        self.fatal_errors = options.get_bool(option_section, "fatal_errors", False)
        self.get_kernel_peaks = options.get_bool(option_section, "get_kernel_peaks", False)
        self.save_kernels = options.get_bool(option_section, "save_kernels", False)

        # Do we want to get the growth rate f(z) from previous datablock calculations
        # or  by taking a spline derivative of tabulated P(k,z)?
        self.fz_from_block = options.get_bool(option_section, "fz_from_block", False)

        self.limber_ell_start = options.get_int(option_section, "limber_ell_start", 300)
        do_exact_string = options.get_string(option_section, "do_exact", "")
        if do_exact_string=="":
            self.do_exact_option_names=[]
        else:
            self.do_exact_option_names = (do_exact_string.strip()).split(" ")
        auto_only_string = options.get_string(option_section, "auto_only", "")
        if auto_only_string=="":
            self.auto_only_option_names=[]
        else:
            self.auto_only_option_names = (auto_only_string.strip()).split(" ")

        self.clip_chi_kernels = options.get_double(option_section, "clip_chi_kernels", 1.e-6)

        # accuracy settings
        self.sig_over_dchi = options.get_double(option_section, "sig_over_dchi", 50.)
        self.shear_kernel_dchi = options.get_double(option_section, "shear_kernel_dchi", 5.)

        # And the req ell ranges.
        # We have some different options for this.
        # - The simplest is log-spaced floats
        # - We allow for linear ell spacing at low ell and switching to log-spaced
        # at some ell
        # - If we're doing the exact calculation we need integer ells. These
        # go up to limber_ell_start. We take the floor of all the ells
        # below limber_ell_start, and keep only unique ones.

        self.ell = np.array([])
        ell_min_logspaced = options.get_double(option_section,
            "ell_min_logspaced", -1.)
        ell_max_logspaced = options.get_double(option_section,
            "ell_max_logspaced", -1.)
        n_ell_logspaced = options.get_int(option_section, "n_ell_logspaced",-1)
        if n_ell_logspaced>0:
            assert ell_min_logspaced>0.
            assert ell_max_logspaced>0.
            self.ell = np.logspace(np.log10(ell_min_logspaced),
                np.log10(ell_max_logspaced), n_ell_logspaced)

        # Optionally add some linearly spaced ells at low ell
        ell_min_linspaced = options.get_int(option_section, "ell_min_linspaced",-1)
        ell_max_linspaced = options.get_int(option_section, "ell_max_linspaced", -1)
        n_ell_linspaced = options.get_int(option_section, "n_ell_linspaced", -1)
        if n_ell_linspaced>0:
            assert ell_max_linspaced>0
            linear_ells = np.linspace(ell_min_linspaced, ell_max_linspaced,
                n_ell_linspaced)
            if len(self.ell>0):
                assert linear_ells[-1]<self.ell[0]
                self.ell = np.concatenate( (linear_ells, self.ell) )
            else:
                self.ell = linear_ells
        if len(self.ell) == 0:
            raise ValueError("Ell range mis-specified.  You now must specify "
                "ell_min_logspaced, ell_max_logspaced, n_ell_logspaced "
                "and/or ell_min_linspaced etc.")

        # Sort out ells for exact calculation
        # We set a limber_ell_start and n_ell_exact in the options
        # Do integer and (approximately) log-spaced ells between 0 and
        # limber_ell_start, and always to 1 also.
        # accuracy settings for exact integral
        self.n_ell_exact = 0
        self.dlogchi = options.get_int(option_section, "dlogchi", -1)
        if self.dlogchi < 0:
            self.dlogchi = None
        self.chi_pad_upper = options.get_double(option_section, "chi_pad_upper", 2.)
        self.chi_pad_lower = options.get_double(option_section, "chi_pad_lower", 2.)

        if len(self.do_exact_option_names)>0:
            sig_over_dchi_exact = options.get_double(option_section, "sig_over_dchi_exact", 20.)
            self.exact_ell_max = self.limber_ell_start
            ell_exact = np.floor(self.ell[self.ell < self.limber_ell_start])
            self.ell_limber = self.ell[self.ell >= self.limber_ell_start]
            self.ell_exact = np.unique(ell_exact)
            self.n_ell_exact = len(self.ell_exact)
            self.do_rsd = options.get_bool(option_section, "do_rsd", False)
            self.exact_kwargs = { "sig_over_dchi": sig_over_dchi_exact,
                                  "dlogchi": self.dlogchi,
                                  "chi_pad_lower": self.chi_pad_lower,
                                  "chi_pad_upper": self.chi_pad_upper,
                                  "do_rsd": self.do_rsd }
        else:
            self.exact_kwargs = None

        # Galaxy bias stuff for Lingal* spectra
        self.lin_bias_prefix = options.get_string(option_section,
                                                  "lin_bias_prefix",
                                                  "b")

        # Check which spectra we are requested to calculate
        self.parse_requested_spectra(options)
        print("Will project these spectra into 2D:")
        for spectrum in self.req_spectra:
            print("    - ", spectrum.section_name)
            if spectrum.section_name in self.do_exact_section_names:
                print(f"Doing exact calculation for {self.n_ell_exact} ells")
                print(f"between {self.ell_exact.min()} and {self.ell_exact.max()}")

        self.kernels = {}
        self.power = {}
        self.outputs = {}

    def parse_requested_spectra(self, options):
        # Get the list of spectra that we want to compute.
        # List of Spectrum objects that we need to compute
        self.req_spectra = []

        # List of keys which determine which kernels are required. These
        # are tuples of (kernel_type, sample_name), where kernel_type is
        # "N" or "W" (for number density and lensing respectively), and
        # sample name determines the n(z) section i.e. nz_<sample_name>.
        self.req_kernel_keys = set()

        # List of options that determines which 3d power spectra
        # are required for the spectra. These are tuples of
        # (PowerType, suffix, do_exact) where
        # - PowerType is a Power3d or child object
        # - suffix is a suffix for the section name.
        # - do_exact is a bool, whether or not we're doing
        # the exact calclation and thus need to load in extra stuff
        self.req_power_options = set()
        self.do_exact_section_names = []
        self.auto_only_section_names = []

        any_spectra_option_found = False
        for spectrum in self.spectrumType:

            spectrum = spectrum.value
            option_name = spectrum.option_name()
            option_name = option_name.replace("density", "position")
            try:
                value = options[option_section, option_name]
            # if value is not set at all, skip
            except BlockError:
                continue

            # If we find any of the options set then record that
            any_spectra_option_found = True


            if isinstance(value, bool):
                if value:
                    raise ValueError(f"""You now need to provide an argument
                        sample_a-sample_b for which to calculate
                        spectrum {option_name}, not just T/F""")
                continue

            # There are various ways a user can describe the spectra:
            #   string of form  euclid-lsst
            #   (one pair of n(z) samples to correlate,
            #   output in the default section)
            #   euclid-lsst:cross  euclid-euclid:auto
            #   (one or more pairs, output in the named section)

            # Otherwise it must be a string - enforce this.
            if not isinstance(value, str):
                raise ValueError(f"Unknown form of value for option {option_name} "
                                 f"in project_2d: {value}")

            value = value.strip()
            if not value:
                raise ValueError(f"Empty value for option {option_name} in project_2d")

            # now we are looking for things of the form
            # shear-shear = euclid-ska[:name]
            # where we would now search for nz_euclid and nz_ska
            values = value.split()
            for value in values:
                try:
                    sample_name_a, sample_name_b = value.split('-', 1)
                    # Optionally we can also name the spectrum, for example
                    # shear-shear = ska-ska:radio
                    # in which case the result will be saved into shear_cl_radio
                    # instead of just shear_cl.
                    # This will be necessary in the case that we run multiple spectra,
                    # e.g.
                    # # shear-shear = euclid-ska:cross  euclid-euclid:optical  ska-ska:radio

                    # would be needed to avoid clashes.
                    # Can also allow
                    # # intrinsic-intrinsic = des_source-des_source:des_power:des_cl kids_source-kids_source:kids_power:kids_cl
                    # to use the suffix XXX or YYY on the IA and galaxy density 3D power spectrum inputs
                    if ":" in sample_name_b:
                        sample_name_b, save_name = sample_name_b.split(":",1)
                        if ":" in save_name:
                            power_suffix, save_name = save_name.split(':',1)
                            power_suffix = "_"+power_suffix
                        else:
                            power_suffix = ""
                    else:
                        save_name = ""
                        power_suffix = ""

                    if power_suffix.startswith("_{") and power_suffix.endswith("}"):
                        suffix_range = power_suffix[2:-1].split("-")
                        suffices = [f"_{x}" for x in range(int(suffix_range[0]), int(suffix_range[1])+1)]
                        only_bins_list = [[i] for i in range(int(suffix_range[0]), int(suffix_range[1])+1)]
                    else:
                        suffices = [power_suffix]
                        only_bins_list = [None]

                    sample_name_a = sample_name_a.strip()
                    sample_name_b = sample_name_b.strip()
                    kernel_key_a = (spectrum.kernel_types[0], sample_name_a)
                    kernel_key_b = (spectrum.kernel_types[1], sample_name_b)
                    self.req_kernel_keys.add(kernel_key_a)
                    self.req_kernel_keys.add(kernel_key_b)

                    for power_suffix, only_bins in zip(suffices, only_bins_list):

                        # power_key is the power_3d class and suffix
                        power_key = (spectrum.power_3d_type, power_suffix)

                        # The self in the line below is not a mistake - the source objects
                        # for the spectrum class is the SpectrumCalculator itself
                        s = spectrum(self, sample_name_a, sample_name_b, power_key, save_name, only_bins=only_bins, len_only_bins=len(only_bins_list))
                        self.req_spectra.append(s)

                        if option_name in self.do_exact_option_names:
                            print("doing exact for option_name", option_name)
                            power_options = power_key + (True,)
                            # It may be that the same power_key, but with do_exact=False is already
                            # in self.req_power_keys. If this is the case remove it. We don't need it.
                            # It's dead to us.
                            try:
                                self.req_power_options.remove(power_key + (False,))
                            except KeyError:
                                pass
                            self.do_exact_section_names.append(s.section_name)
                            self.req_power_options.add(power_options)
                        else:
                            power_options = power_key + (False,)
                            if (power_key+(True,)) not in self.req_power_options:
                                self.req_power_options.add(power_options)

                        if option_name in self.auto_only_option_names:
                            self.auto_only_section_names.append(s.section_name)

                        print("Calculating Limber: "
                            f"Kernel 1 = {kernel_key_a}, "
                            f"Kernel 2 = {kernel_key_b}, "
                            f"P_3D = {power_key[0].__name__}, "
                            f"section = {power_key[0].get_section_name(power_key[1])} "
                            f"--> Output: {s.section_name}")
                except:
                    raise ValueError("To specify a P(k)->C_ell projection with one or "
                                     "more sets of two different n(z) samples use the "
                                     "form shear-shear=sample1-sample2 sample3-sample4 ...")

        # If no other spectra are specified, just do the shear-shear spectrum.
        if not any_spectra_option_found:
            print("")
            print("No spectra requested in the parameter file.")
            print("I will go along with this and just do nothing,")
            print("but if you get a crash later this is probably why.")
            print("")

    def load_distance_splines(self, block):
        # Extract some useful distance splines
        z_distance = block[names.distances, 'z']
        a_distance = block[names.distances, 'a']
        chi_distance = block[names.distances, 'd_m']

        # have to copy these to get into C ordering (because we reverse them),
        # otherwise we get a view on the original array.
        if z_distance[1] < z_distance[0]:
            z_distance = z_distance[::-1].copy()
            a_distance = a_distance[::-1].copy()
            chi_distance = chi_distance[::-1].copy()

        h0 = block[names.cosmological_parameters, "h0"]

        # convert Mpc to Mpc/h
        chi_distance *= h0

        # This is only needed by CMB spectra
        if block.has_value(names.distances, 'CHISTAR'):
            self.chi_star = block[names.distances, 'CHISTAR'] * h0
        else:
            self.chi_star = None

        self.chi_max = chi_distance.max()
        self.a_of_chi = interp.InterpolatedUnivariateSpline(chi_distance, a_distance)
        self.chi_of_z = interp.InterpolatedUnivariateSpline(z_distance, chi_distance)
        self.max_z_for_chi = z_distance.max()
        self.dchidz = self.chi_of_z.derivative()
        self.chi_distance = chi_distance

    def load_kernels(self, block):
        # During the setup we already decided what kernels (W(z) or N(z) splines)
        # we needed for the spectra we want to do. Load them now and add to the
        # self.kernels dictionary.
        for key in self.req_kernel_keys:
            kernel_type, sample_name = key

            # For all the sources we need the n(z)
            if (sample_name not in self.kernels) and (sample_name != 'cmb'):
                section_name = "nz_"+sample_name
                self.kernels[sample_name] = TomoNzKernel.from_block(block, section_name, norm=True)
            if (sample_name not in self.kernels) and (sample_name == 'cmb'):
                zmin = 0.001
                zmax = block['distances','zstar']
                z_arr_for_cmb = np.exp(np.linspace(np.log(zmin), np.log(zmax), num = 1000))
                self.kernels[sample_name] =  TomoNzKernel(z_arr_for_cmb, 0, is_cmb_lensing = True)
                
            if kernel_type == "N":
                self.kernels[sample_name].set_nofchi_splines(self.chi_of_z, self.dchidz,
                    clip=self.clip_chi_kernels)

            elif kernel_type == "W":
                self.kernels[sample_name].set_wofchi_splines(self.chi_of_z,
                    self.dchidz, self.a_of_chi, clip=self.clip_chi_kernels,
                    dchi=self.shear_kernel_dchi)

            elif kernel_type == 'K' and sample_name == 'cmb':
                if self.max_z_for_chi < 1000.0:
                    raise ValueError("To get CMB lensing you need to compute distances to very high z. Consider setting the zmax_logz and n_logz parameters if using CAMB.")
                chi_star = block['distances','chistar']
                h0 = block[names.cosmological_parameters, "h0"]
                self.kernels[sample_name].set_cmblensing_splines(self.chi_of_z, self.a_of_chi, chi_star*h0, clip = self.clip_chi_kernels)

            elif kernel_type == "W_W":
                self.kernels[sample_name].set_wwofchi_splines(self.chi_of_z,
                    self.dchidz, self.a_of_chi, clip=self.clip_chi_kernels,
                    dchi=self.shear_kernel_dchi)

            elif kernel_type == "F":
                # This is the combined shear and IA kernel. We need to calculate
                # F(z) = - A *(z/z0)**alpha * C1_RHOCRIT * Omega_m / growth(z)
                # This kernel is only valid for simple IA models.
                ia_params = "intrinsic_alignment_parameters"
                a_ia = block[ia_params, "A"]
                alpha = block.get_double(ia_params, "alpha", 0.)
                z0 = block.get_double(ia_params, "z0", 0.5)

                # Get the growth rate from the matter power, by finding
                # the nearest k value to 1e-3
                z,k,pk_lin = block.get_grid(names.matter_power_lin, "z", "k_h", "p_k")
                k_growth = 1.e-3
                growth_ind = np.argmin(np.abs(k-k_growth))
                growth_vals = np.sqrt(pk_lin[:, growth_ind] / pk_lin[0, growth_ind])

                omega_m = block[names.cosmological_parameters, "omega_m"]

                F_of_z = (-1 * a_ia * C1_RHOCRIT * ((1+z)/(1+z0))**alpha
                    * omega_m / growth_vals)

                F_of_chi_spline = interp.InterpolatedUnivariateSpline(
                    self.chi_of_z(z), F_of_z)

                self.kernels[sample_name].set_combined_shear_ia_splines(
                    self.chi_of_z, self.dchidz, self.a_of_chi, F_of_chi_spline,
                    self.lensing_prefactor, clip=self.clip_chi_kernels,
                    dchi=self.shear_kernel_dchi)
            else:
                raise ValueError(f"Invalid kernel type: {kernel_type}. Should be 'N' or 'W'")

    def save_kernels_to_block(self, block):
        for name, kernel in self.kernels.items():
            section = f"kernels_{name}"
            kernel.to_block(block, section)

    def load_power(self, block):
        # Loop through keys in self.req_power_keys, initializing the Power3d
        # instances and adding to the self.power dictionary.
        for power_options in self.req_power_options:
            powertype, suffix, do_exact = power_options
            power = powertype(suffix)
            if self.verbose:
                print(f"Loading {power.section_name} 3D power spectrum")
            power.load_from_block(block, self.chi_of_z)
            if do_exact:
                power.set_nonlimber_splines(block, self.chi_of_z, fz_from_block = self.fz_from_block)
            power_key = (powertype, suffix)
            self.power[power_key] = power

    def load_lensing_prefactor(self, block):
        self.lensing_prefactor = get_lensing_prefactor(block)

    def load_lensing_weyl_prefactor(self, block):
        self.lensing_weyl_prefactor = get_lensing_weyl_prefactor(block)


    def compute_spectrum(self, block, spectrum):

        # Save some naming about the spectrum
        sep_name = "ell"
        block[spectrum.section_name, "save_name"] = spectrum.save_name
        block[spectrum.section_name, "sample_a"] = spectrum.sample_a
        block[spectrum.section_name, "sample_b"] = spectrum.sample_b
        block[spectrum.section_name, "sep_name"] = sep_name

        # And bin count info
        na, nb = spectrum.nbins()
        if spectrum.is_autocorrelation():
            block[spectrum.section_name, 'nbin'] = na
        block[spectrum.section_name, 'nbin_a'] = na
        block[spectrum.section_name, 'nbin_b'] = nb

        # Save is_auto (whether this is an autocorrelation)
        block[spectrum.section_name, 'is_auto'] = spectrum.is_autocorrelation()

        # Save auto_only - whether we've only computed autocorrelations
        block[spectrum.section_name, 'auto_only'] = (
            spectrum.section_name in self.auto_only_section_names)

        # Set up any required power splines
        spectrum.prepare(block, lin_bias_prefix=self.lin_bias_prefix)

        if self.verbose:
            print(f"Computing spectrum {spectrum.__class__.__name__} ({spectrum.section_name}) for samples"
                  f" ({spectrum.sample_a}, {spectrum.sample_b}) from P(k) {spectrum.input_section_name}")

        # The caching of P(k) at the grid points we need is more effective if
        # everything shares a chimin and chimax so set that now, even though it
        # means that we are doing some integration over a wider range than strictly
        # necessary for some bin pairs.
        chimin, chimax, dchi = spectrum.compute_shared_chi_range(self.sig_over_dchi)

        for i in range(na):
            if not spectrum.should_do_bin(i+1) and spectrum.len_only_bins == na:
                continue
            # for auto-correlations C_ij = C_ji so we calculate only one of them,
            # but save both orderings to the block to account for different ordering
            # conventions.
            # for cross-correlations we must do both
            jmax = i+1 if spectrum.is_autocorrelation() else nb
            for j in range(jmax):
                if spectrum.section_name in self.auto_only_section_names:
                    if j!=i:
                        continue
                if not spectrum.should_do_bin(j+1) and spectrum.kernel_types[0] == spectrum.kernel_types[1]:
                    continue

                if self.verbose:
                    print(f"    Computing bin pair {i+1}, {j+1}")
                # For some (user-chosen) spectra we compute the non-limber
                # spectrum for a subset of the ell range
                if spectrum.section_name in self.do_exact_section_names:
                    exact_kwargs = self.exact_kwargs.copy()
                    if spectrum.has_rsd:
                        exact_kwargs["do_rsd"] = self.do_rsd
                    ell, c_ell = spectrum.compute(block, self.ell_limber, i+1, j+1,
                        ell_exact=self.ell_exact,
                        chimin=chimin, chimax=chimax, dchi_limber=dchi,
                        exact_kwargs=exact_kwargs
                    )
                else:
                    ell, c_ell = spectrum.compute(block, self.ell, i+1, j+1,
                                                  chimin=chimin, chimax=chimax, dchi_limber=dchi,
                    )

                block[spectrum.section_name, sep_name] = ell
                block[spectrum.section_name, f'bin_{i+1}_{j+1}'] = c_ell

    def clean(self):
        # need to manually delete power spectra we have loaded
        self.power.clear()

        # spectra know how to delete themselves, in gsl_wrappers.py
        self.kernels.clear()
        self.outputs.clear()
        self.lensing_prefactor = None

        self.interpolation_cache.clear()

    def execute(self, block):
        """
        Execute the pipeline step by loading the relevant data from the block
        (distance splines, kernels, P(k)s) and compute the C(l)s.
        """
        try:
            # Load input information from the block
            self.interpolation_cache = {}
            self.load_distance_splines(block)
            self.load_lensing_prefactor(block)
            self.load_lensing_weyl_prefactor(block)

            # Loading the kernels can often fail, so we catch that specifically
            # and explain the causes
            t0 = timer()
            self.load_kernels(block)
            t1 = timer()
            if self.verbose:
                t = timedelta(seconds=t1-t0)
                print(f"Time to set up kernels: {t}")

            # We can optionally save the kernels that go into the
            # integration as well.  This is useful for e.g. plotting
            # things.  
            if self.save_kernels:
                self.save_kernels_to_block(block)

            # Load the P(k)s
            t0 = timer()
            self.load_power(block)

            t1 = timer()
            if self.verbose:
                t = timedelta(seconds=t1-t0)
                print(f"Time to load power: {t}")

            # Now loop through the required spectra calling the compute function
            for spectrum in self.req_spectra:
                # For optimization it is useful to individually report the time
                # taken for each set of spectra
                t0 = timer()

                self.compute_spectrum(block, spectrum)

                t1 = timer()
                if self.verbose:
                    t = timedelta(seconds=(t1-t0))
                    print(f"Time to compute spectrum {spectrum.section_name}: {t}")
        finally:
            self.clean()
        return 0

def setup(options):
    return SpectrumCalculator(options)


def execute(block, config):
    return config.execute(block)

def document_spectrum_types():
    # for generating the module.yaml content
    for x in project_2d.SpectrumType:
        x = x.value
        opt_name = x.option_name()
        sec = x.power_3d_type.section
        name = x.name
        print(f"{opt_name:34}{sec:34}{name}")
