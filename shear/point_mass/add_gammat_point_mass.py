import os
import sys
current_dir = os.path.dirname(__file__)
import ctypes as ct
import scipy as sp
path = os.path.join(current_dir, '../../structure/projection/projection_tools')
sys.path.append(path)
from gsl_wrappers import GSLSpline
from cosmosis.datablock import names
from cosmosis.datablock import option_section
from scipy.interpolate import InterpolatedUnivariateSpline as IUSpline
import numpy as np
import scipy.integrate
import astropy.units as u
import astropy.constants as const
try:
    from scipy.integrate import simpson
except ImportError:
    from scipy.integrate import simps as simpson

sigcrit_inv_fac = (4 * np.pi * const.G)/(const.c**2)
sigcrit_inv_fac_Mpc_Msun = (sigcrit_inv_fac.to(u.Mpc/u.M_sun)).value

def get_Dcom_array(zarray, Omega_m, H0, Omega_L=None):
    if Omega_L is None:
        Omega_L = 1. - Omega_m
    c = 299792.458
    Dcom_array = np.zeros(len(zarray))
    for j in range(len(zarray)):
        zf = zarray[j]
        res1 = sp.integrate.quad(lambda z: (c / H0) * (1 / (np.sqrt(Omega_L + Omega_m * ((1 + z) ** 3)))), 0, zf)
        Dcom = res1[0]
        Dcom_array[j] = Dcom
    return Dcom_array




def setup(options):
    config = {}

    # Section name that has the free variables names
    config['B_section'] = options.get_string(option_section, "bias_section", "add_pm")
    config['B_prefix'] = options.get_string(option_section, "bias_prefix", "b_")
    config['add_togammat'] = options.get_bool(option_section, 'add_togammat',False)
    config['use_fiducial'] = options.get_bool(option_section, 'use_fiducial',False)

    config['Omega_m_fid'] = options.get_double(option_section, 'Omega_m_fid',0.3)
    config['H0_fid'] = options.get_double(option_section, 'H0_fid',69.0)

    # lens n(z) section
    config["lens_nz_section"] = options.get_string(option_section, "lens_nz_section", "nz_lens")
    config["source_nz_section"] = options.get_string(option_section, "source_nz_section", "nz_source")

    # theory and output section
    config["gammat_section"] = options.get_string(option_section, "gammat_section", "galaxy_shear_xi")
    config["sigcrit_inv_section"] = options.get_string(option_section, "sigcrit_inv_section", "sigma_crit_inv_lens_source")

    if config['use_fiducial']:
        z_distance = np.linspace(0.0,6.2,1000)
        chi_distance = get_Dcom_array(z_distance, config['Omega_m_fid'], config['H0_fid'])
        chi_of_z = GSLSpline(z_distance, chi_distance * (config['H0_fid']/100.) )
        config["chi_of_z"] = chi_of_z
        config['betaj1j2'] = {}


    return config


def execute(block, config):

    if config['use_fiducial']:
        chi_of_z = config["chi_of_z"]
    else:
        h0 = block[names.cosmological_parameters, "hubble"] / 100.
        Omega_m = block[names.cosmological_parameters, "omega_m"]

        z_distance = block[names.distances, 'z']
        chi_distance = block[names.distances, 'd_m']
        if z_distance[1]<z_distance[0]:
            z_distance = z_distance[::-1].copy()
            chi_distance = chi_distance[::-1].copy()

        chi_of_z = GSLSpline(z_distance, chi_distance * h0)

    num_lens_z = block[config['lens_nz_section'], 'nbin']
    num_source_z = block[config['source_nz_section'], 'nbin']
    z_lens = block[config['lens_nz_section'], 'z'][1:]
    z_source = block[config['source_nz_section'], 'z'][1:]

    if (config['use_fiducial'] and ((str(0) + '_' + str(0)) not in list(config['betaj1j2']))) or (not config['use_fiducial']):
        # Setup the variables needed for sigma_crit_inverse
        chi_lens = chi_of_z(z_lens)
        chi_source = chi_of_z(z_source)
    
        chi_lmat = np.tile(chi_lens.reshape(len(z_lens), 1), (1, len(z_source)))
        chi_smat = np.tile(chi_source.reshape(1, len(z_source)), (len(z_lens), 1))
        num = chi_smat - chi_lmat
        ind_lzero = np.where(num <= 0)
        num[ind_lzero] = 0

    if config['add_togammat']:
        theta_th = block['galaxy_shear_xi','theta']




    for j1 in range(num_lens_z):
        if config['add_togammat']:
            Bj1 = block[config['B_section'], config['B_prefix'] + str(j1 + 1)]
        for j2 in range(num_source_z):
            if (config['use_fiducial'] and ((str(j1) + '_' + str(j2)) not in list(config['betaj1j2']))) or (not config['use_fiducial']):
                nz_lens = block[config['lens_nz_section'], "bin_%d" % (j1 + 1)][1:]
                nz_source = block[config['source_nz_section'], "bin_%d" % (j2 + 1)][1:]

                ng_array_source_rep = np.tile(nz_source.reshape(1, len(z_source)), (len(z_lens), 1))
                int_sourcez = simpson(ng_array_source_rep * (num / chi_smat), x=z_source)

                coeff_ints = sigcrit_inv_fac_Mpc_Msun

                Is = coeff_ints * chi_lens * (1. + z_lens) * int_sourcez

                # Evaluate the Cij of Eq 24 written in the comoving coordinates
                # Since gamma_t is a scalar it should be same in both physical and comoving coordinates
                # It is just easier to match the expressions in comoving coordinates to the ones on methods paper.
                
                betaj1j2_pm = simpson(nz_lens * Is * (1./chi_lens**2), x=z_lens)
                if (config['use_fiducial']):
                    config['betaj1j2'][str(j1) + '_' + str(j2)] = betaj1j2_pm
            else:
                betaj1j2_pm = config['betaj1j2'][str(j1) + '_' + str(j2)]
            
            if config['add_togammat']:
                Cj1j2_pm = Bj1 * 1e13 * betaj1j2_pm

                # Get the already evaluated gamma_t from theory for each bin
                gt_j1j2 = block[config["gammat_section"], "bin_" + str(j1 + 1) + '_' + str(j2 + 1)]

                # Add the point mass term as evaluated above
                gt_addpm = gt_j1j2 + Cj1j2_pm/(theta_th ** 2)

                # Write it back to the theory section
                block[config["gammat_section"], "bin_" + str(j1 + 1) + '_' + str(j2 + 1)] = gt_addpm
                block['point_mass',"bin_" + str(j1 + 1) + '_' + str(j2 + 1)] = Cj1j2_pm

            block[config["sigcrit_inv_section"],"sigma_crit_inv_%d_%d"%(j1+1, j2+1)] = 1e13 * betaj1j2_pm

    return 0
