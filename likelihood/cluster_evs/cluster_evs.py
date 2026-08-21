import numpy as np
import os
from cosmosis.datablock import names as section_names
from cosmosis.datablock import option_section
import scipy.integrate
from math import log10, floor
from scipy.interpolate import UnivariateSpline, RectBivariateSpline, SmoothBivariateSpline

cosmo = section_names.cosmological_parameters
clusters = section_names.clusters
likes = section_names.likelihoods
evs = section_names.evs
mf = section_names.mass_function
dist = section_names.distances


def setup(options):
    section = option_section
    feedback = options.get_int(section, "feedback", default=0)
    redshift = options.get_double(section, "redshift", default=1.)
    output_pdf = options.get_bool(section, "output_pdf", default=False)
    frac = options.get_double(section, "frac", default=1.0)
    Mmin = options.get_double(section, "M_min", default=1.e14)
    Mmax = options.get_double(section, "M_max", default=2.e15)
    n_m = options.get_int(section, "n_M", default=100)
    minput = np.logspace(np.log10(Mmin), np.log10(Mmax), n_m)
    return (feedback, redshift, frac, minput, output_pdf)


def massfunction(m, zz, rbs):
    z = round(zz, 2)
    # This spline can be produce Nan or negative values for fm below if the mf has been saved on a coarse grid. soln is to increase nsteps in massfunction module
    return rbs.ev(m, z)


def dndmint(logm, zz, rbs):
    m = np.exp(logm)
    integrand = massfunction(m, zz, rbs)
    return integrand[0]


def dvdm_zint(zz, m, omega_matter, h0, interp_da, rbs):
    return (dVcdz(zz, omega_matter, h0, interp_da) * (1. / m) * massfunction(m, zz, rbs))[0]


def dvdzdndmint(zz, Mmin, Mmax, omega_matter, h0, interp_da, rbs):
    Mint = scipy.integrate.quad(dndmint, np.log(Mmin), np.log(
        Mmax), args=(zz, rbs), epsrel=1e-6, epsabs=0)[0]
    return Mint * dVcdz(zz, omega_matter, h0, interp_da)


def dVcdz(z, omega_matter, h0, interp_da):
    c_light = 3.e5  # km/s
    da_z = interp_da(z)
    # (Mpc/h)^3
    return c_light / (h0 * 100.0) * (da_z**2) * (1.0 + z)**2 / (np.sqrt(omega_matter * (1.0 + z)**3 + (1.0 - omega_matter))) * (h0**3)


def execute(block, config):
    # Configuration data, read from ini file above
    feedback, redshift, frac, minput, output_pdf = config
    zmin = redshift - 0.01
    zmax = redshift + 0.01

    # The maximum true mass of the cluster
    maxmass = block[clusters, 'M_max']

    # Cosmological distances
    omega_matter = block[cosmo, 'omega_m']
    h0 = block[cosmo, 'h0']
    z_da = block[dist, "z"]
    da_array = block[dist, "d_a"]
    interp_da = UnivariateSpline(z_da, da_array)

    # Mass function
    zarray = block[mf, "z"]
    rarray = block[mf, "R_H"]
    rho_m = 2.775e11 * (omega_matter)  # h^2 M_solar Mpc^-3.
    marray = (4.0 * 3.1415 / 3.0) * rho_m * rarray**3  # M_solar/h
    dndmarray = block[mf, "dndlnMh"].reshape(
        [np.size(zarray), np.size(marray)]).T
    Mmin = 1.e12
    Mmax = 1.e18
    # 2D interpolator into mass function
    rbs = RectBivariateSpline(marray, zarray, dndmarray)

    ntot = scipy.integrate.quad(dvdzdndmint, zmin, zmax, args=(
        Mmin, Mmax, omega_matter, h0, interp_da, rbs), epsrel=1e-6, epsabs=0)[0]
    NUM = ntot * frac

    LogPhi = np.zeros(minput.size)
    i = 0
    # Optionally (switched on in the ini file; this takes a bit longer)
    # Generate a complete PDF of the maximum mass instead of just at the
    # specified maximum mass
    if output_pdf:
        for i, mm in enumerate(minput):
            FFm = 1.0 / ntot * scipy.integrate.quad(
                dvdzdndmint, zmin, zmax,
                args=(Mmin, mm, omega_matter, h0, interp_da, rbs),
                epsrel=1e-6, epsabs=0)[0]
            fm = 1.0 / ntot * (scipy.integrate.quad(
                dvdm_zint, zmin, zmax,
                args=(mm, omega_matter, h0, interp_da, rbs),
                epsrel=1e-6, epsabs=0)[0])
            LogPhi[i] = np.log(NUM * fm) + (NUM - 1) * np.log(FFm)

        # Save the complete PDF
        block[evs, 'logphi'] = LogPhi
        block[evs, 'm'] = minput

    # Always genereate the log-likelihood of M_max
    FFm = 1.0 / ntot * scipy.integrate.quad(dvdzdndmint, zmin, zmax, args=(
        Mmin, maxmass, omega_matter, h0, interp_da, rbs), epsrel=1e-6, epsabs=0)[0]

    fm = 1.0 / ntot * (scipy.integrate.quad(dvdm_zint, zmin, zmax, args=(
        maxmass, omega_matter, h0, interp_da, rbs), epsrel=1e-6, epsabs=0)[0])

    LogLike = np.log(NUM * fm) + (NUM - 1) * np.log(FFm)

    # Save the likelihood
    block[likes, 'EVS_LIKE'] = LogLike

    # signal that everything went fine
    return 0


def cleanup(config):
    return 0
