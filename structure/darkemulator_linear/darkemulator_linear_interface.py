"""
Author: Sunao Sugiyama
Last edit: 2023.05.16
See Nishimichi et al. (https://arxiv.org/pdf/1811.09504.pdf) for detail.

Code reviewed by Tianqing Zhang in Oct 2023
"""
import sys
import os
from cosmosis.datablock import names, option_section
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy import integrate
import numpy as np
import astropy.cosmology

def setup(options):
    from dark_emulator.darkemu.pklin import pklin_gp
    pklin_emulator = pklin_gp()
    
    more_config = {}
    # redshift: linear matter power spectrum grids
    z_spacings = options.get_string(option_section, 'z_space').split()
    z_edges    = options.get_double_array_1d(option_section, 'z_edge')
    z_mins     = z_edges[:-1]
    z_maxs     = z_edges[1:] - 1e-4 # to avoid overlapping 
    z_nbins    = options.get_int_array_1d(option_section, 'z_nbin')
    z = []
    for z_spacing, z_min, z_max, z_nbin in zip(z_spacings, z_mins, z_maxs, z_nbins):
        if z_spacing == 'log':
            zsub = np.logspace(np.log10(z_min), np.log10(z_max), z_nbin)
        elif z_spacing == 'lin':
            zsub = np.linspace(z_min, z_max, z_nbin)
        z.append(zsub)
    more_config['z'] = np.hstack(z)
    # more_config['zmin']   = options.get_double(option_section, 'zmin', default=1e-2)
    # more_config['zmid']   = options.get_double(option_section, 'zmid', default=0.1)
    # more_config['zmax']   = options.get_double(option_section, 'zmax', default=10.0)
    # more_config['nztot']  = options.get_int(option_section, 'nztot', default=350)
    # more_config['nzmid']  = options.get_int(option_section, 'nzmid', default=50)
    # fourier mode: linear matter power spectrum grids 
    more_config['kmin'] = options.get_double(option_section, "kmin", default=1e-4)
    more_config['kmax'] = options.get_double(option_section, "kmax", default=1e3)
    more_config['nk']   = options.get_int(option_section, "nk", default=500)
    
    # background
    more_config['zmin_background'] = options.get_double(option_section, 'zmin_background', default=z_mins[0])
    more_config['zmax_background'] = options.get_double(option_section, 'zmax_background', default=z_maxs[-1])
    more_config['nz_background'] = options.get_int(option_section, 'nz_background', default=300)
    
    return [pklin_emulator, more_config]

def execute(block, config):
    pklin_emulator, more_config = config
    
    # Get cosmological parameters
    pars = names.cosmological_parameters
    ombh2      = block[pars, 'ombh2']
    h          = block[pars, "h0"]
    log1e10As  = block[pars, "log1e10As"]
    omch2      = block[pars, "omch2"]
    ns         = block[pars, "n_s"]
    w0         = block[pars, "w"]
    mnu   = block[pars, "mnu"]
    omnuh2     = 0.00064 * (mnu/0.06)
    Omm        = (ombh2 + omch2 + omnuh2)/h**2
    Omk   = block[pars, "omega_k"]
    Omde       = 1.0-Omm-Omk
    wa    = block[pars, "wa"]
    
    # test compatibility
    if wa != 0.0:
        raise "dark emulator does not support cosmology with wa != 0."
    if mnu!= 0.06: 
        raise "dark emulator does not support cosmology with mnu != 0.06"
    if Omk != 0.0:
        raise "dark emulator does not support cosmology with Omega_k != 0.0"
    
    # Set derived cosmological parameters to emulator cosmology utils
    from dark_emulator.darkemu import cosmo_util
    cosmo = cosmo_util.cosmo_class()
    params= np.array([ombh2, omch2, Omde, log1e10As, ns, w0])
    cosmo_util.test_cosm_range_linear(params)
    cosmo.cparam = params.reshape((1,6))
    pklin_emulator.set_cosmology(cosmo)
    block[pars, 'omega_m'] = Omm
    block[pars, 'omega_lambda'] = Omde
    block[pars, 'omnuh2'] = omnuh2
    # Compute
    # Dark-emulator computes the linear matter power spectrum in the factorized form:
    #   P_lin(k, z) = D_growth(z) P(k, z=0)
    z = more_config['z']
    k = np.logspace(np.log10(more_config['kmin']), np.log10(more_config['kmax']), more_config['nk'])
    Dg = get_Dgrowth_from_z(cosmo, z)
    P0 = pklin_emulator.get(k)
    P0_2d, Dg_2d = np.meshgrid(P0, Dg)
    pklin_table = P0_2d*Dg_2d**2
    
    # Save 
    block.put_grid(names.matter_power_lin, "z", z, "k_h", k, "p_k", pklin_table)
    
    # cosmological parameters
    sigma8_0 = get_sigmaR(k, P0, 8.0)
    block[names.cosmological_parameters, "sigma_8"] = sigma8_0
    
    # Save growth
    block[names.growth_parameters, "z"] = z
    block[names.growth_parameters, "a"] = 1/(1+z)
    block[names.growth_parameters, "sigma_8"] = sigma8_0 * Dg**2
    # block[names.growth_parameters, "fsigma_8"] = fsigma_8
    # block[names.growth_parameters, "rs_DV"] = rs_DV
    # block[names.growth_parameters, "H"] = H
    # block[names.growth_parameters, "DA"] = DA
    # block[names.growth_parameters, "F_AP"] = F_AP
    block[names.growth_parameters, "d_z"] = Dg
    block[names.growth_parameters, "f_z"] = get_fgrowth_from_z(cosmo, z)
    
    # sigma12 and S_8 - other variants of sigma_8
    sigma12_0 = get_sigmaR(k, P0, 12.0)
    block[names.cosmological_parameters, "sigma_12"] = sigma12_0
    block[names.cosmological_parameters, "S_8"] = sigma8_0*np.sqrt(Omm/0.3)

    # Distance
    z_background = np.linspace(more_config["zmin_background"], 
                               more_config["zmax_background"], 
                               more_config["nz_background"])
    
    block[names.distances, "nz"] = len(z_background)
    block[names.distances, "z"] = z_background
    block[names.distances, "a"] = 1/(z_background+1)
    
    c = astropy.cosmology.w0waCDM(h*100, Omm, Omde, Ob0=ombh2/h**2, w0=w0, wa=wa)
    D_C = c.comoving_distance(z_background).value # Mpc
    H = c.H(z_background).value * 1e3 / 299792458.0
    D_H = 1 / H
    
    if Omk == 0:
        D_M = D_C
    elif Omk < 0:
        s = np.sqrt(-Omk)
        D_M = (D_H / s)  * np.sin(s * D_C / D_H)
    else:
        s = np.sqrt(Omk)
        D_M = (D_H / s) * np.sinh(s * D_C / D_H)

    D_L = D_M * (1 + z_background)
    D_A = D_M / (1 + z_background)
    D_V = ((1 + z_background)**2 * z_background * D_A**2 / H)**(1./3.)
    
    block[names.distances, "D_C"] = D_C # Note that this is in unit of Mpc, not in Mpc/h which is usually used in weak lensing analysis
    block[names.distances, "D_M"] = D_M
    block[names.distances, "D_L"] = D_L
    block[names.distances, "D_A"] = D_A
    block[names.distances, "D_V"] = D_V
    block[names.distances, "H"] = H
    # block[names.distances, "MU"] = mu
    
    
    return 0

# UTILITIES
# returns growth on a given z array
def get_Dgrowth_from_z(cosmo, z):
    # print(z)
    if isinstance(z, (int, float)):
        return cosmo.Dgrowth_from_z(z)
    elif z.size > 900:
        z_arr = np.linspace(z.min(), z.max(), 100)
        Dp = np.array([cosmo.Dgrowth_from_z(_z) for _z in z_arr])
        return ius(z_arr, Dp)(z)
    else:
        return np.array([cosmo.Dgrowth_from_z(_z) for _z in z])

def get_fgrowth_from_z(cosmo, z, dz=0.001):
    # zarr= np.array([z-dz, z+dz])
    # lna = -np.log(1+zarr)
    # lnD = np.log(get_Dgrowth_from_z(cosmo, zarr))
    # return (lnD[1,:] - lnD[0,:])/(lna[1,:] - lna[0,:])

    zarr_p = z+dz
    zarr_m = z-dz
    lna_p = -np.log(1+zarr_p)
    lna_m = -np.log(1+zarr_m)
    
    lnD_p = np.log(get_Dgrowth_from_z(cosmo, zarr_p))
    lnD_m = np.log(get_Dgrowth_from_z(cosmo, zarr_m))
    return (lnD_p - lnD_m)/(lna_p - lna_m)

# Returns spline
def get_z2D(cosmo, zmin=0, zmax=15, nbin=200):
    z = np.linspace(zmin, zmax, nbin)
    D = get_Dgrowth_from_z(cosmo, z)
    z2D = ius(z,D, ext=3)
    return z2D

def get_z2fz(cosmo, zmin=0, zmax=15, nbin=200, dz=0.001):
    z2D = get_z2D(cosmo, zmin=zmin-dz, zmax=zmax+dz)
    z  = np.linspace(zmin, zmax, nbin)
    zp = z-0.0001
    zm = z+0.0001
    lnDp = np.log(z2D(zp))
    lnDm = np.log(z2D(zm))
    lnap = -np.log(1+zp)
    lnam = -np.log(1+zm)
    fz = (lnDp-lnDm)/(lnap-lnam)
    z2fz = ius(z, fz, ext=3)
    return z2fz

# sigma8
def _window_tophat(kR):
    return 3.*(np.sin(kR)-kR*np.cos(kR))/kR**3

def get_sigmaR(k, pk, R, kmin=1e-4, kmax=1e3, nk=500):
    ks = np.logspace(np.log10(kmin), np.log10(kmax), nk)
    logks = np.log(ks)
    kR = ks * R
    integrant = ks**3* ius(k, pk)(ks) * _window_tophat(kR)**2
    return np.sqrt(integrate.trapz(integrant, logks)/(2.*np.pi**2))

