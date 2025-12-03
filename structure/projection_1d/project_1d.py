"""
This module combines tomographic / stellar mass bins of the individually calculated observables.
It produces the theoretical prediction for the observable for the full survey.
The number of bins and the mass range can be different to what is calculated in the hod_interface.py module.

Furthermore it corrects the individually calculated observables (stellar mass function)
for the difference in input data cosmology to the predicted output cosmology
by multiplication of ratio of volumes according to More et al. 2013 and More et al. 2015
"""

import ast
from cosmosis.datablock import option_section, names
import numpy as np
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline
from scipy.integrate import simpson
from astropy.cosmology import Flatw0waCDM
import astropy.cosmology as cosmology_classes

class TomoNzKernel(object):
    def __init__(self, z, nzs, norm=True):
        self.z = z
        self.nzs = {}
        self.nbin = 0
        for i,nz in enumerate(nzs):
            self.nbin += 1
            if norm:
                nz_spline = InterpolatedUnivariateSpline(self.z, nz)
                norm = nz_spline.integral(self.z[0], self.z[-1])
                nz = nz/norm
                self.nzs[i+1] = nz

    @classmethod
    def from_block(cls, block, section_name, norm=True):
        nbin = block[section_name, "nbin"]
        z = block[section_name, "z"]
        nzs = []
        for i in range(1, nbin+1):
            nz = block[section_name, "bin_%d"%i]
            nzs.append(nz)
        return cls(z, nzs, norm=norm)
    
    @property
    def interpolated_nz(self):
        nz_interp = []
        for i in range(self.nbin):
            nz_interp.append(interp1d(self.z, self.nzs[i+1], kind='linear', fill_value=0.0, bounds_error=False))
        return nz_interp

def load_and_interpolate_obs(block, obs_section, suffix_in):
    """
    Loadsthe observable function, e.g. stellar mass function,
    and the redshift bins for the observable. Interpolates the observable function for the redshift values
    that are given.
    """
    # Load observable values from observable section name, suffix_in is either med for median
    # or a number showing the observable-redshift bin index
    obs_in = block[obs_section, f'obs_val{suffix_in}']
    obs_func_in = block[obs_section, f'obs_func{suffix_in}']

    # If there are any observable-redshift bins in the observable section:
    # If there are no bins z_bin_{suffix_in} does not exist
    if block.has_value(obs_section, f'z_bin{suffix_in}'):
        z_obs = block[obs_section, f'z_bin{suffix_in}']
        if len(z_obs) == 1: # We have zmed per bin
            obs_func_interp = lambda x: obs_func_in
            z_obs = None
        else:
            obs_func_interp = interp1d(
                z_obs, obs_func_in, kind='linear',
                fill_value='extrapolate', bounds_error=False, axis=0
            )
    else:
        obs_func_interp = lambda x: obs_func_in
        z_obs = None
    return z_obs, obs_in, obs_func_interp

def setup(options):
    config = {}

    # Input and output section names
    input_section_name = options.get_string(option_section, 'input_section_name', default='stellar_mass_function')
    config['output_section_name'] = options.get_string(option_section, 'output_section_name', default='smf')
    config['correct_cosmo'] = options.get_bool(option_section, 'correct_cosmo', default=False)
    config['observable_type'] = options.get_string(option_section, 'observable_type', default='mass')
    if config['observable_type'] not in ['mass', 'luminosity']:
        raise ValueError('Currently supported observable types are mass or luminosity (with observable function being either stellar mass function or luminosity function).')

    if ":" in input_section_name:
        input_section_name, suffix = input_section_name.split(':',1)
    else:
        suffix = ""
    config['input_section_name'] = input_section_name

    if suffix.startswith("{") and suffix.endswith("}"):
        suffix_range = suffix[1:-1].split("-")
        config['suffixes'] = [f"_{x}" for x in range(int(suffix_range[0]), int(suffix_range[1])+1)]
    else:
        config['suffixes'] = [f"_{suffix}"]
    
    config['nbins'] = len(config['suffixes'])
    config['sample'] = options.get_string(option_section, 'sample', '')
    
    if config['correct_cosmo']:
        config['zmin'] = np.asarray([options[option_section, 'zmin']]).flatten()
        config['zmax'] = np.asarray([options[option_section, 'zmax']]).flatten()
        # Check if the length of zmin, zmax, nbins match
        if len(config['zmin']) != config['nbins'] or len(config['zmax']) != config['nbins']:
            raise ValueError('Error: zmin, zmax need to be of the same length as the number of bins provided.')
        # cosmo_kwargs is to be a string containing a dictionary with all the arguments the
        # requested cosmology accepts (see default)!
        cosmo_kwargs = ast.literal_eval(
            options.get_string(
                option_section, 'cosmo_kwargs', default="{'H0':70.0, 'Om0':0.3, 'Ode0':0.7}"
            )
        )
        # Requested cosmology class from astropy:
        cosmo_class = options.get_string(
            option_section, 'astropy_cosmology_class', default='LambdaCDM'
        )
        cosmo_class_init = getattr(cosmology_classes, cosmo_class)
        cosmo_model_data = cosmo_class_init(**cosmo_kwargs)
    
        config['cosmo_model_data'] = cosmo_model_data
        config['h_data'] = cosmo_model_data.h
            
    return config

def execute(block, config):
    input_section_name = config['input_section_name']
    output_section_name = config['output_section_name']
    suffixes = config['suffixes']
    nbins = config['nbins']

    try:
        nz = TomoNzKernel.from_block(block, config['sample'], norm=True)
    except:
        nz = None

    if config['correct_cosmo']:
        zmin = config['zmin']
        zmax = config['zmax']
        h_data = config['h_data']
        cosmo_model_data = config['cosmo_model_data']

        # Adopting the same cosmology object as in halo_model_ingredients module
        tcmb = block.get_double(names.cosmological_parameters, 'TCMB', default=2.7255)
        cosmo_model_run = Flatw0waCDM(
            H0=block[names.cosmological_parameters, 'hubble'],
            Ob0=block[names.cosmological_parameters, 'omega_b'],
            Om0=block[names.cosmological_parametersarams, 'omega_m'],
            m_nu=[0, 0, block[names.cosmological_parameters, 'mnu']],
            Tcmb0=tcmb, w0=block[names.cosmological_parameters, 'w'],
            wa=block[names.cosmological_parameters, 'wa']
        )
        h_run = cosmo_model_run.h

    for i in range(nbins):
        # Reads in and produce the interpolator for obs_func. z_obs is read if it exists.
        z_obs, obs_arr, obs_func_interp = load_and_interpolate_obs(block, input_section_name, suffixes[i])

        if z_obs is not None:
            z = block.get_double_array_1d(names.distances, 'z')
            nz_inter = nz.interpolated_nz[i]
            obs_func = simpson(nz_inter(z)[:, np.newaxis] * obs_func_interp(z), z, axis=0) 
        else:
            obs_func = obs_func_interp(1)

        if config['correct_cosmo']:
            obs_func_in = obs_func.copy()
            comoving_volume_data = ((cosmo_model_data.comoving_distance(zmax[i])**3.0
                                 - cosmo_model_data.comoving_distance(zmin[i])**3.0)
                                * h_data**3.0)
            comoving_volume_model = ((cosmo_model_run.comoving_distance(zmax[i])**3.0
                                  - cosmo_model_run.comoving_distance(zmin[i])**3.0)
                                 * h_run**3.0)

            ratio_obs = comoving_volume_model / comoving_volume_data
            obs_func = obs_func_in * ratio_obs

        block.put_double_array_1d(output_section_name, f'bin_{i + 1}', np.squeeze(obs_func))
        block.put_double_array_1d(output_section_name, f'{config["observable_type"]}_{i + 1}', np.squeeze(obs_arr))
    block[output_section_name, 'nbin'] = nbins
    block[output_section_name, 'sample'] = config['sample'] if config['sample'] is not None else 'None'

    return 0

def cleanup(config):
    # Usually python modules do not need to do anything here.
    # We just leave it in out of pedantic completeness.
    pass
