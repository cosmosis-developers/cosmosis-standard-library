import numpy as np
import copy
import pickle
import os

from numpy.core.shape_base import atleast_1d
from scipy.sparse import coo
from .utils import _transform_space, MyProgressBar, mean_absolute_exp_percentage_error, accuracy_exp_002, accuracy_exp_005

import tensorflow
tensorflow.compat.v1.logging.set_verbosity(tensorflow.compat.v1.logging.ERROR)
from tensorflow.keras.models import load_model
from .baryonic_boost import load_baryonic_emu
from scipy import interpolate
gpus = tensorflow.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tensorflow.config.experimental.set_memory_growth(gpu, True)

__all__ = ["Matter_powerspectrum"]

class Matter_powerspectrum(object):
    """
    A class to load and call the baccoemu for the matter powerspectrum.
    By default, the linear power spectrum (described in Aricò et al. 2021), the nonlinear boost
    (described in Angulo et al. 2020), and the baryonic boost (described in Aricò et al. 2020c) are loaded.

    The emulators for A_s, sigma8, sigma12, and the smearing and removing of the BAO are also available.

    At large scales, i.e. k<0.01, the ratios between non-linear/linear and baryonic/non-linear spectra is considered to be 1.

    :param linear: whether to load the linear emulator, defaults to True
    :type linear: boolean, optional
    :param smeared_bao: whether to load the smeared-BAO emulator, defaults to False
    :type smeared_bao: boolean, optional
    :param no_wiggles: whether to load the no-wiggles emulator, defaults to True
    :type no_wiggles: boolean, optional
    :param nonlinear_boost: whether to load the nonlinear boost emulator, defaults to True
    :type nonlinear_boost: boolean, optional
    :param baryonic_boost: whether to load the baryonic boost emulator, defaults to True
    :type baryonic_boost: boolean, optional
    :param compute_sigma8: whether to load the sigma8 emulator, defaults to True
    :type compute_sigma8: boolean, optional
    :param verbose: whether to activate the verbose mode, defaults to True
    :type verbose: boolean, optional

    """
    def __init__(self, linear=True, smeared_bao=False, no_wiggles=True, nonlinear_boost = True,
                 baryonic_boost=True, compute_sigma8=True,
                 nonlinear_emu_path=None, nonlinear_emu_details=None,
                 baryonic_emu_path=None, baryonic_emu_details=None, verbose=True):

        self.verbose = verbose

        self.compute_linear = True if linear else False
        self.compute_smeared_bao = True if smeared_bao else False
        self.compute_no_wiggles = True if no_wiggles else False
        self.compute_nonlinear_boost = True if nonlinear_boost else False
        self.compute_baryonic_boost = True if baryonic_boost else False
        self.compute_sigma8 = True if compute_sigma8 else False

        if self.compute_smeared_bao and self.compute_no_wiggles:
            print("""Provide only one between the smeared BAO and the non-wiggles emulators!""")
            raise ValueError("""Set either compute_smeared_bao=False (default) or no_wiggles=False!""")

        self.emulator = {}
        if self.compute_sigma8:
            self.emulator['sigma8']  = load_sigma8_emu(verbose=verbose)

        if self.compute_linear:
            self.emulator['linear'] = load_linear_emu(verbose=verbose)

        if self.compute_smeared_bao:
            self.emulator['smeared_bao'] = load_smeared_bao_emu(verbose=verbose)

        if self.compute_no_wiggles:
            self.emulator['no_wiggles'] = load_no_wiggles_emu(verbose=verbose)

        if self.compute_nonlinear_boost:
            self.emulator['nonlinear'] = load_nonlinear_emu(fold_name=nonlinear_emu_path, detail_name=nonlinear_emu_details, verbose=verbose)

        if self.compute_baryonic_boost:
            self.emulator['baryon'] = load_baryonic_emu(fold_name=baryonic_emu_path, detail_name=baryonic_emu_details, verbose=verbose)

    def _get_parameters(self, coordinates, which_emu):
        """
        Function that returns a dictionary of cosmological parameters,
        computing derived cosmological parameters, if not
        already present in the given coordinates, and checking the relevant boundaries.
        :param coordinates: a set of coordinates in parameter space
        :type coordinates: dict
        :param which_emu: kind of emulator: options are 'linear', 'nonlinear','baryon','smeared_bao','sigma8'
        :type which_emu: str
        :return: coordinates with derived parameters
        :rtype: dict
        """
        coordinates = {key: np.atleast_1d(coordinates[key]) for key in set(list(coordinates.keys())) - set(['k', 'k_lin', 'pk_lin'])}

        avail_pars = [coo for coo in coordinates.keys() if coordinates[coo][0] is not None] #parameters currently available
        eva_pars = self.emulator[which_emu]['keys']  #parameters strictly needed to evaluate the emulator
        req_pars = self.emulator[which_emu]['keys'] if which_emu != 'linear' else self.emulator[which_emu]['full_keys'] #parameters needed for a computation
        comp_pars = list(set(req_pars)-set(avail_pars)) #parameters to be computed
        deriv_pars = ['omega_cold','sigma8_cold', 'A_s'] #derived parameters that can be computed
        miss_pars = list(set(comp_pars)-set(deriv_pars)) #parameters missing from coordinates
        extra_pars = list(set(req_pars)-set(eva_pars)) #requested parameters not needed for evaluation
        if miss_pars:
            print(f"{which_emu} emulator:")
            print(f"  Please add the parameter(s) {miss_pars} to your coordinates!")
            raise KeyError(f"{which_emu} emulator: coordinates need the following parameters: ", miss_pars)

        if ('omega_cold' in avail_pars) & ('omega_matter' in avail_pars):
            assert len(coordinates['omega_cold']) == len(coordinates['omega_matter']), 'Both omega_cold and omega_matter were provided, but they have different len'
            om_from_oc = coordinates['omega_cold'] + coordinates['neutrino_mass'] / 93.14 /coordinates['hubble']**2
            assert np.all(np.abs(coordinates['omega_matter'] - om_from_oc) < 1e-4), 'Both omega_cold and omega_matter were provided, but they are inconsistent among each other'

        if 'omega_cold' in comp_pars:
            if 'omega_matter' not in avail_pars:
                raise KeyError('One parameter between omega_matter and omega_cold must be provided!')

            omega_nu = coordinates['neutrino_mass'] / 93.14 /coordinates['hubble']**2
            coordinates['omega_cold'] = coordinates['omega_matter'] - omega_nu

        if ('sigma8_cold' not in avail_pars) & ('A_s' not in avail_pars):
            raise KeyError('One parameter between sigma8_cold and A_s must be provided!')

        if ('sigma8_cold' in  avail_pars) & ('A_s' in avail_pars):
            #commented for the cases where one is computed and same value is repeated
            #assert len(np.atleast_1d(coordinates['sigma8_cold'])) == len(atleast_1d(coordinates['A_s'])), 'Both sigma8_cold and A_s were provided, but they have different len'

            ignore_s8_pars = copy.deepcopy(coordinates)
            del ignore_s8_pars['sigma8_cold']
            ignore_s8_pars['cold'] = True
            s8_from_A_s = self.get_sigma8(**ignore_s8_pars)
            assert np.all(np.abs(coordinates['sigma8_cold'] - s8_from_A_s) < 1e-4), 'Both sigma8_cold and A_s were provided, but they are inconsistent among each other'

        if 'sigma8_cold' in comp_pars:
            tmp_coords = copy.deepcopy(coordinates)
            tmp_coords['cold']=True
            coordinates['sigma8_cold'] = np.atleast_1d(self.get_sigma8(**tmp_coords))

        if 'A_s' in comp_pars:
            tmp_coords = copy.deepcopy(coordinates)
            del tmp_coords['sigma8_cold']
            tmp_coords['A_s'] = 2e-9
            tmp_coords['cold'] = True
            coordinates['A_s'] = np.atleast_1d((coordinates['sigma8_cold'] / self.get_sigma8(**tmp_coords))**2 * tmp_coords['A_s'])

        pp = np.squeeze([coordinates[p][0] for p in eva_pars])
        coords_out = copy.deepcopy(coordinates)

        grid = {}
        for key in coordinates.keys():
            if len(np.atleast_1d(coordinates[key])) > 1:
                grid[key] = np.array(coordinates[key])

        if len(list(grid.keys()))==0:
            grid = None
        else:
            grid_structure = []
            for key in grid.keys():
                grid_structure.append(len(grid[key]))
            grid_structure = np.array(grid_structure)
            values, counts = np.unique(grid_structure, return_counts=True)
            counts_but_highest = np.delete(counts, np.argmax(counts))
            assert np.all(counts == counts[0]) | np.all(counts_but_highest == 1), 'When passing multiple coordinate sets you should either vary only on parameter, or all parameters should have the same len'

        if grid is not None:
            grid_pars = list(grid.keys()) # list of parameters that are varyied in a grid
            N = len(grid[grid_pars[0]])
            pp = np.tile(pp, (N, 1))
            for par in grid_pars:
                if par in eva_pars:
                    index = eva_pars.index(par)
                    pp[:,index] = np.float64(grid[par])
                if par in req_pars:
                    coords_out[par] = grid[par]
            pp = np.float64(pp)

        for i,par in enumerate(eva_pars):
            val = pp[i] if grid is None else pp[:,i]
            message = 'Param {}={} out of bounds [{}, {}]'.format(
                par, val, self.emulator[which_emu]['bounds'][i][0], self.emulator[which_emu]['bounds'][i][1])

            assert np.all(val >= self.emulator[which_emu]['bounds'][i][0]) & np.all(val <= self.emulator[which_emu]['bounds'][i][1]), message

        if extra_pars:
            cc = np.squeeze([coords_out[p] for p in extra_pars])
            if None in cc:
                raise ValueError(f'None in parameters: {extra_pars} = {cc}!')

        return coords_out, pp, grid


    def _evaluate_nonlinear(self, **kwargs):
        """Evaluate the given emulator at a set of coordinates in parameter space.

        The coordinates must be specified as a dictionary with the following
        keywords:

        #. 'omega_cold'
        #. 'omega_baryon'
        #. 'sigma8_cold'
        #. 'hubble'
        #. 'ns'
        #. 'neutrino_mass'
        #. 'w0'
        #. 'wa'
        #. 'expfactor'
        #. 'k' : a vector of wavemodes in h/Mpc at which the nonlinear boost will be computed, if None
                 the default wavemodes of the nonlinear emulator will be used, defaults to None
        #. 'k_lin': a vector of wavemodes in h/Mpc, if None the wavemodes used by
                    the linear emulator are returned, defaults to None
        #. 'pk_lin': a vector of linear matter power spectrum computed at k_lin, either cold or total depending
                     on the key "cold". If None the linear emulator will be called, defaults to None
        #. 'cold': whether to return the cold matter power spectrum or the total one. Default to True
        """
        if not self.compute_nonlinear_boost:
            raise ValueError("Please enable the nonlinear boost!")

        k = kwargs['k'] if 'k' in kwargs.keys() else None
        k_lin = kwargs['k_lin'] if 'k_lin' in kwargs.keys() else None
        pk_lin = kwargs['pk_lin'] if 'pk_lin' in kwargs.keys() else None
        cold = kwargs['cold'] if 'cold' in kwargs.keys() else True

        emulator = self.emulator['nonlinear']

        coords, pp, grid = self._get_parameters(kwargs, 'nonlinear')

        _pp = _transform_space(pp, space_rotation=False, bounds=emulator['bounds'])

        yrec = emulator['model'](_pp.reshape(-1,9), training=False)
        Q = np.squeeze(np.exp(emulator['scaler'].inverse_transform(yrec)))

        if pk_lin is None:
            pk_lin_kwargs = copy.deepcopy(kwargs)
            pk_lin_kwargs['k'] = None
            k_lin, pk_lin = self.get_linear_pk(**pk_lin_kwargs)
        else:
            k_lin = np.squeeze(k_lin)
            pk_lin = np.squeeze(pk_lin)
            if k_lin is None:
                raise ValueError("""If the linear power spectrum pk_lin is provided,
                                    also the wavenumbers k_lin at which is computed must be provided """)
            elif (np.amin(k_lin)>1e-3) | (np.amax(k_lin) < 10):
                raise ValueError(f"""
                    A minimum k <= 0.001 h/Mpc and a maximum k >= 10 h/Mpc
                    are required in the linear power spectrum for the calculation
                    of the non linear boost:
                    the current values are {np.amin(k_lin)}) h/Mpc and {np.amax(k_lin)} h/Mpc
                    """)
        if cold:
            k_lin_cold = k_lin
            pk_lin_cold = pk_lin
        else:
            k_lin_tot = k_lin
            pk_lin_tot = pk_lin
            total_kwargs = copy.deepcopy(kwargs)
            total_kwargs['cold'] = True
            k_lin_cold, pk_lin_cold = self.get_linear_pk(**total_kwargs)

        pklin_interp_cold = interpolate.interp1d(np.log(k_lin_cold), np.log(pk_lin_cold),
                kind='linear', axis = -1 if grid is None else 1, fill_value='extrapolate')
        pk_lin_emu_cold = np.exp(pklin_interp_cold(np.log(emulator['k'])))

        if not cold:
            pklin_interp_tot = interpolate.interp1d(np.log(k_lin_tot), np.log(pk_lin_tot),
                    kind='linear', axis = -1 if grid is None else 1, fill_value='extrapolate')
            pk_lin_emu_tot = np.exp(pklin_interp_tot(np.log(emulator['k'])))
            pk_lin_emu = pk_lin_emu_tot
        else:
            pk_lin_emu = pk_lin_emu_cold

        if self.compute_smeared_bao:
            smeared_bao_kwargs = copy.deepcopy(kwargs)
            smeared_bao_kwargs['k'] = emulator['k']
            _, pk_smeared = self.get_smeared_bao_pk(**smeared_bao_kwargs)

        elif self.compute_no_wiggles:
            no_wiggles_kwargs = copy.deepcopy(kwargs)
            no_wiggles_kwargs['k'] = emulator['k']
            no_wiggles_kwargs['k_lin'] = k_lin_cold
            no_wiggles_kwargs['pk_lin'] = pk_lin_cold
            _, pk_no_wiggles = self.get_no_wiggles_pk(**no_wiggles_kwargs)
            pk_smeared = _smeared_bao_pk(k_lin=k_lin_cold, pk_lin=pk_lin_cold, k_emu=emulator['k'], pk_lin_emu=pk_lin_emu_cold, pk_nw=pk_no_wiggles, grid=grid)
        else:
            pk_smeared = _smeared_bao_pk(k_lin=k_lin_cold, pk_lin=pk_lin_cold, k_emu=emulator['k'], pk_lin_emu=pk_lin_emu_cold, grid=grid)

        if cold:
            nonlinear_boost = Q * pk_smeared / pk_lin_emu_cold
        else:
            omega_nu = kwargs['neutrino_mass'] / 93.14 / kwargs['hubble']**2
            omega_matter = kwargs['omega_cold'] + omega_nu if kwargs['omega_cold'] is not None else kwargs['omega_matter']
            f_nu = omega_nu/omega_matter
            if grid is None:
                add = pk_lin_emu_tot - (1-f_nu)**2 * pk_lin_emu_cold
            else:
                add = pk_lin_emu_tot - ((1-f_nu)**2 * pk_lin_emu_cold.T).T

            if grid is None:
                nonlinear_boost = (Q*pk_smeared*(1-f_nu)**2 + add)/pk_lin_emu_tot
            else:
                nonlinear_boost = (((Q*pk_smeared).T*(1-f_nu)**2).T + add)/pk_lin_emu_tot

        if k is not None:
            k_max = max(k)
            kemu_max = self.emulator['nonlinear']['k'].max()
            kemu_min = self.emulator['nonlinear']['k'].min()

            if (k_max > kemu_max):
                raise ValueError(f"""The nonlinear emulator must be {kemu_max} h/Mpc: the current value is {k_max} h/Mpc""")

            nl_interp = interpolate.interp1d(np.log(emulator['k']), nonlinear_boost,
                                             kind='linear', axis = -1 if grid is None else 1, fill_value='extrapolate', bounds_error=False,
                                             assume_sorted=True)
            nonlinear_boost = np.squeeze(nl_interp(np.log(k)))
            if grid is None:
                nonlinear_boost[k<kemu_min] = 1.
            else:
                nonlinear_boost[:,k<kemu_min] = 1.

            pklin_interp = interpolate.interp1d(np.log(k_lin), np.log(pk_lin),
            kind='linear', axis = -1 if grid is None else 1, fill_value='extrapolate',
            assume_sorted=True)
            pk_lin_emu = np.exp(pklin_interp(np.log(k)))
        else :
            k = emulator['k']

        return  k, nonlinear_boost, np.squeeze(nonlinear_boost*pk_lin_emu)

    def get_nonlinear_boost(self, omega_cold=None, omega_matter=None, omega_baryon=None,
                            sigma8_cold=None, A_s=None, hubble=None, ns=None, neutrino_mass=None,
                            w0=None, wa=None, expfactor=None, k=None, k_lin=None, pk_lin=None, cold=True, **kwargs):
        """Compute the prediction of the nonlinear boost of cold matter power spectrum

        :param omega_cold: omega cold matter (cdm + baryons), either omega_cold
                           or omega_matter should be specified, if both are specified
                           they should be consistent
        :type omega_cold: float or array
        :param omega_matter: omega total matter (cdm + baryons + neutrinos), either omega_cold
                           or omega_matter should be specified, if both are specified
                           they should be consistent
        :type omega_matter: float or array
        :param sigma8_cold: rms of cold (cdm + baryons) linear perturbations, either sigma8_cold
                            or A_s should be specified, if both are specified they should be
                            consistent
        :type sigma8_cold: float or array
        :param A_s: primordial scalar amplitude at k=0.05 1/Mpc, either sigma8_cold
                    or A_s should be specified, if both are specified they should be
                    consistent
        :type A_s: float or array
        :param hubble: adimensional Hubble parameters, h=H0/(100 km/s/Mpc)
        :type hubble: float or array
        :param ns: scalar spectral index
        :type ns: float or array
        :param neutrino_mass: total neutrino mass
        :type neutrino_mass: float or array
        :param w0: dark energy equation of state redshift 0 parameter
        :type w0: float or array
        :param wa: dark energy equation of state redshift dependent parameter
        :type wa: float or array
        :param expfactor: expansion factor a = 1 / (1 + z)
        :type expfactor: float or array
        :param k: a vector of wavemodes in h/Mpc at which the nonlinear boost will be computed, if None
                the default wavemodes of the nonlinear emulator will be used, defaults to None
        :type k: array_like, optional
        :param k_lin: a vector of wavemodes in h/Mpc, if None the wavemodes used by
                  the linear emulator are returned, defaults to None
        :type k_lin: array_like, optional
        :param pk_lin: a vector of linear power spectrum computed at k_lin, if None
                  the linear emulator will be called, defaults to None
        :type pk_lin: array_like, optional
        :param cold: whether to return the cold matter power spectrum or the total one. Default to cold.
        :type cold: bool, optional
        :return: k and Q(k), the emulated nonlinear boost of P(k)
        :rtype: tuple
        """
        _kwargs = locals()
        kwargs = {key: _kwargs[key] for key in set(list(_kwargs.keys())) - set(['self'])}
        value = self._evaluate_nonlinear(**kwargs)
        return value[0], value[1]

    def get_baryonic_boost(self, omega_cold=None, omega_matter=None, omega_baryon=None,
                           sigma8_cold=None, A_s=None, hubble=None, ns=None, neutrino_mass=None,
                           w0=None, wa=None, expfactor=None, M_c=None, eta=None, beta=None,
                           M1_z0_cen=None, theta_out=None, theta_inn=None, M_inn=None,
                           k=None, **kwargs):
        """Evaluate the baryonic emulator at a set of coordinates in parameter space.

        :param omega_cold: omega cold matter (cdm + baryons), either omega_cold
                           or omega_matter should be specified, if both are specified
                           they should be consistent
        :type omega_cold: float or array
        :param omega_matter: omega total matter (cdm + baryons + neutrinos), either omega_cold
                           or omega_matter should be specified, if both are specified
                           they should be consistent
        :type omega_matter: float or array
        :param sigma8_cold: rms of cold (cdm + baryons) linear perturbations, either sigma8_cold
                            or A_s should be specified, if both are specified they should be
                            consistent
        :type sigma8_cold: float or array
        :param A_s: primordial scalar amplitude at k=0.05 1/Mpc, either sigma8_cold
                    or A_s should be specified, if both are specified they should be
                    consistent
        :type A_s: float or array
        :param hubble: adimensional Hubble parameters, h=H0/(100 km/s/Mpc)
        :type hubble: float or array
        :param ns: scalar spectral index
        :type ns: float or array
        :param neutrino_mass: total neutrino mass
        :type neutrino_mass: float or array
        :param w0: dark energy equation of state redshift 0 parameter
        :type w0: float or array
        :param wa: dark energy equation of state redshift dependent parameter
        :type wa: float or array
        :param expfactor: expansion factor a = 1 / (1 + z)
        :type expfactor: float or array
        :param M_c: mass fraction of hot gas in haloes
        :type M_c: float or array
        :param eta: extent of ejected gas
        :type eta: float or array
        :param beta: mass fraction of hot gas in haloes
        :type beta: float or array
        :param M1_z0_cen: characteristic halo mass scale for central galaxy
        :type M1_z0_cen: float or array
        :param theta_out: density profile of hot gas in haloes
        :type theta_out: float or array
        :param theta_inn: density profile of hot gas in haloes
        :type theta_inn: float or array
        :param M_inn: density profile of hot gas in haloes
        :type M_inn: float or array
        :param k: a vector of wavemodes in h/Mpc at which the nonlinear boost will be computed, if None
                the default wavemodes of the baryonic emulator will be used, defaults to None
        :type k: array_like, optional
        :param grid: dictionary with parameter and vector of values where to evaluate the emulator, defaults to None
        :type grid: array_like, optional
        :return: k and S(k), the emulated baryonic boost of P(k)
        :rtype: tuple
        """
        _kwargs = locals()
        kwargs = {key: _kwargs[key] for key in set(list(_kwargs.keys())) - set(['self'])}

        if not self.compute_baryonic_boost:
            raise ValueError("Please enable the baryonic boost!")

        emulator = self.emulator['baryon']
        coords, pp, grid = self._get_parameters(kwargs, 'baryon')

        _pp = _transform_space(np.array([pp]), space_rotation=False, bounds=emulator['bounds'])
        yrec = emulator['model'](_pp.reshape(-1,len(emulator['keys'])), training=False)
        baryonic_boost = np.squeeze(np.exp(emulator['scaler'].inverse_transform(yrec)))

        if k is not None:
            k_max = max(k)
            kemu_max = emulator['k'].max()
            kemu_min = emulator['k'].min()
            if (k_max > kemu_max) & (self.verbose):
                print(f""" WARNING:
                The maximum k of the baryonic boost emulator is {kemu_max} h/Mpc:
                the baryonic emulator is currently extrapolating to {k_max} h/Mpc;
                the extrapolation will likely be not accurate.
                """)

            baryonic_interp = interpolate.interp1d(np.log(emulator['k']), baryonic_boost,
                                                   kind='linear', axis = 0 if grid is None else 1, fill_value='extrapolate', bounds_error=False,
                                                   assume_sorted=True)
            baryonic_boost = baryonic_interp(np.log(k))
            if grid is None:
                baryonic_boost[k<kemu_min] = 1.
            else:
                baryonic_boost[:,k<kemu_min] = 1.
        else :
            k = emulator['k']

        return  k, baryonic_boost

    def get_nonlinear_pk(self, omega_cold=None, omega_matter=None, omega_baryon=None,
                         sigma8_cold=None, A_s=None, hubble=None, ns=None, neutrino_mass=None,
                         w0=None, wa=None, expfactor=None, M_c=None, eta=None, beta=None,
                         M1_z0_cen=None, theta_out=None, theta_inn=None, M_inn=None,
                         k=None, k_lin=None, pk_lin=None, cold=True, baryonic_boost=False, **kwargs):
        """Compute the prediction of the nonlinear cold matter power spectrum.

        :param omega_cold: omega cold matter (cdm + baryons), either omega_cold
                           or omega_matter should be specified, if both are specified
                           they should be consistent
        :type omega_cold: float or array
        :param omega_matter: omega total matter (cdm + baryons + neutrinos), either omega_cold
                           or omega_matter should be specified, if both are specified
                           they should be consistent
        :type omega_matter: float or array
        :param sigma8_cold: rms of cold (cdm + baryons) linear perturbations, either sigma8_cold
                            or A_s should be specified, if both are specified they should be
                            consistent
        :type sigma8_cold: float or array
        :param A_s: primordial scalar amplitude at k=0.05 1/Mpc, either sigma8_cold
                    or A_s should be specified, if both are specified they should be
                    consistent
        :type A_s: float or array
        :param hubble: adimensional Hubble parameters, h=H0/(100 km/s/Mpc)
        :type hubble: float or array
        :param ns: scalar spectral index
        :type ns: float or array
        :param neutrino_mass: total neutrino mass
        :type neutrino_mass: float or array
        :param w0: dark energy equation of state redshift 0 parameter
        :type w0: float or array
        :param wa: dark energy equation of state redshift dependent parameter
        :type wa: float or array
        :param expfactor: expansion factor a = 1 / (1 + z)
        :type expfactor: float or array
        :param M_c: mass fraction of hot gas in haloes
        :type M_c: float or array
        :param eta: extent of ejected gas
        :type eta: float or array
        :param beta: mass fraction of hot gas in haloes
        :type beta: float or array
        :param M1_z0_cen: characteristic halo mass scale for central galaxy
        :type M1_z0_cen: float or array
        :param theta_out: density profile of hot gas in haloes
        :type theta_out: float or array
        :param theta_inn: density profile of hot gas in haloes
        :type theta_inn: float or array
        :param M_inn: density profile of hot gas in haloes
        :type M_inn: float or array
        :param k: a vector of wavemodes in h/Mpc at which the nonlinear boost will be computed, if None the default wavemodes of the emulator will be used, defaults to None
        :type k: array_like, optional
        :param k_lin: a vector of wavemodes in h/Mpc, if None the wavemodes used by
                  the linear emulator are returned, defaults to None
        :type k_lin: array_like, optional
        :param pk_lin: a vector of linear power spectrum computed at k_lin, if None
                  the linear emulator will be called, defaults to None
        :type pk_lin: array_like, optional
        :param cold: whether to return the cold matter power spectrum or the total one. Default to cold.
        :type cold: bool, optional
        :return: k and the nonlinear P(k)
        :rtype: tuple
        """
        _kwargs = locals()
        kwargs = {key: _kwargs[key] for key in set(list(_kwargs.keys())) - set(['self'])}

        k, nl_boost, pk_nl = self._evaluate_nonlinear(**kwargs)

        if baryonic_boost:
            bb_kwargs = copy.deepcopy(kwargs)
            bb_kwargs['k'] = k
            k, baryon_boost = self.get_baryonic_boost(**bb_kwargs)
        else:
            baryon_boost = 1.

        return k, pk_nl*baryon_boost


    def get_linear_pk(self, omega_cold=None, omega_matter=None, omega_baryon=None,
                            sigma8_cold=None, A_s=None, hubble=None, ns=None, neutrino_mass=None,
                            w0=None, wa=None, expfactor=None, k=None, cold=True, **kwargs):
        """Evaluate the linear emulator at a set of coordinates in parameter space.

        :param omega_cold: omega cold matter (cdm + baryons), either omega_cold
                           or omega_matter should be specified, if both are specified
                           they should be consistent
        :type omega_cold: float or array
        :param omega_matter: omega total matter (cdm + baryons + neutrinos), either omega_cold
                           or omega_matter should be specified, if both are specified
                           they should be consistent
        :type omega_matter: float or array
        :param sigma8_cold: rms of cold (cdm + baryons) linear perturbations, either sigma8_cold
                            or A_s should be specified, if both are specified they should be
                            consistent
        :type sigma8_cold: float or array
        :param A_s: primordial scalar amplitude at k=0.05 1/Mpc, either sigma8_cold
                    or A_s should be specified, if both are specified they should be
                    consistent
        :type A_s: float or array
        :param hubble: adimensional Hubble parameters, h=H0/(100 km/s/Mpc)
        :type hubble: float or array
        :param ns: scalar spectral index
        :type ns: float or array
        :param neutrino_mass: total neutrino mass
        :type neutrino_mass: float or array
        :param w0: dark energy equation of state redshift 0 parameter
        :type w0: float or array
        :param wa: dark energy equation of state redshift dependent parameter
        :type wa: float or array
        :param expfactor: expansion factor a = 1 / (1 + z)
        :type expfactor: float or array
        :param k: a vector of wavemodes in h/Mpc at which the nonlinear boost will be computed, if None
                the default wavemodes of the linear emulator will be used, defaults to None
        :type k: array_like, optional
        :param k_lin: a vector of wavemodes in h/Mpc, if None the wavemodes used by
                  the linear emulator are returned, defaults to None
        :type k_lin: array_like, optional
        :param pk_lin: a vector of linear power spectrum computed at k_lin, if None
                  the linear emulator will be called, defaults to None
        :type pk_lin: array_like, optional
        :param cold: whether to return the cold matter power spectrum or the total one. Default to cold.
        :type cold: bool, optional
        :return: k and the linear P(k)
        :rtype: tuple
        """
        _kwargs = locals()
        kwargs = {key: _kwargs[key] for key in set(list(_kwargs.keys())) - set(['self'])}

        if not self.compute_linear:
            raise ValueError("Please enable the linear emulator!")

        emulator = self.emulator['linear']

        coordinates, pp, grid = self._get_parameters(kwargs, 'linear')

        model = 'model_cold' if cold else 'model_tot'
        scaler = 'scaler_cold' if cold else 'scaler_tot'
        ypred = emulator[model](pp.reshape(-1,7), training=False)
        pk_lin = np.squeeze(np.exp(emulator[scaler].inverse_transform(ypred)))

        As_fixed = 1.e-9
        ns_fixed = 1.
        k_pivot = 0.05

        if grid is None:
            As = coordinates['A_s']
            ns = coordinates['ns']
            h = coordinates['hubble']
            kk = emulator['k']
        else:
            As = coordinates['A_s'][:,None] if 'A_s' in grid.keys() else coordinates['A_s']
            ns = coordinates['ns'][:,None] if 'ns' in grid.keys() else coordinates['ns']
            h = coordinates['hubble'][:,None] if 'hubble' in grid.keys() else coordinates['hubble']
            kk = emulator['k'][None,:]

        pk_lin *= As/As_fixed * (kk/k_pivot*h)**(ns-ns_fixed)

        if k is not None:
            if (max(k) > max(emulator['k']))|(min(k) < min(emulator['k'])):
                raise ValueError(f"""
                    A minimum k > {min(emulator['k'])} h/Mpc and a maximum k < {max(emulator['k'])} h/Mpc
                    are required for the linear emulator:
                    the current values are {min(k)} h/Mpc and {max(k)} h/Mpc
                    """)

            else:
                if np.any(pk_lin < 0):
                    pklin_interp = interpolate.interp1d(np.log(emulator['k']), pk_lin,
                                                        kind='linear', axis = 0 if grid is None else 1, fill_value='extrapolate',
                                                        assume_sorted=True)
                    pk_lin = pklin_interp(np.log(k))
                else:
                    pklin_interp = interpolate.interp1d(np.log(emulator['k']), np.log(pk_lin),
                                                        kind='linear', axis = 0 if grid is None else 1, fill_value='extrapolate',
                                                        assume_sorted=True)
                    pk_lin = np.exp(pklin_interp(np.log(k)))
        else:
            k = emulator['k']
        return  k, pk_lin


    def get_smeared_bao_pk(self, omega_cold=None, omega_matter=None, omega_baryon=None,
                           sigma8_cold=None, A_s=None, hubble=None, ns=None, neutrino_mass=None,
                           w0=None, wa=None, expfactor=None, k=None, **kwargs):
        """Evaluate the cold matter power spectrum with smeared bao, calling an emulator
            at a set of coordinates in parameter space.

        :param omega_cold: omega cold matter (cdm + baryons), either omega_cold
                           or omega_matter should be specified, if both are specified
                           they should be consistent
        :type omega_cold: float or array
        :param omega_matter: omega total matter (cdm + baryons + neutrinos), either omega_cold
                           or omega_matter should be specified, if both are specified
                           they should be consistent
        :type omega_matter: float or array
        :param sigma8_cold: rms of cold (cdm + baryons) linear perturbations, either sigma8_cold
                            or A_s should be specified, if both are specified they should be
                            consistent
        :type sigma8_cold: float or array
        :param A_s: primordial scalar amplitude at k=0.05 1/Mpc, either sigma8_cold
                    or A_s should be specified, if both are specified they should be
                    consistent
        :type A_s: float or array
        :param hubble: adimensional Hubble parameters, h=H0/(100 km/s/Mpc)
        :type hubble: float or array
        :param ns: scalar spectral index
        :type ns: float or array
        :param neutrino_mass: total neutrino mass
        :type neutrino_mass: float or array
        :param w0: dark energy equation of state redshift 0 parameter
        :type w0: float or array
        :param wa: dark energy equation of state redshift dependent parameter
        :type wa: float or array
        :param expfactor: expansion factor a = 1 / (1 + z)
        :type expfactor: float or array
        :param k: a vector of wavemodes in h/Mpc at which the nonlinear boost will be computed, if None
                the default wavemodes of the linear emulator will be used, defaults to None
        :type k: array_like, optional
        :param k_lin: a vector of wavemodes in h/Mpc, if None the wavemodes used by
                  the linear emulator are returned, defaults to None
        :type k_lin: array_like, optional
        :param pk_lin: a vector of linear power spectrum computed at k_lin, if None
                  the linear emulator will be called, defaults to None
        :type pk_lin: array_like, optional
        :param grid: dictionary with parameters and vectors of values where to evaluate the emulator, defaults to None
        :type grid: array_like, optional

        :return: k and the linear P(k)
        :rtype: tuple
        """

        _kwargs = locals()
        kwargs = {key: _kwargs[key] for key in set(list(_kwargs.keys())) - set(['self'])}

        if not self.compute_smeared_bao:
            raise ValueError("Please enable the smeared bao emulator!")

        emulator = self.emulator['smeared_bao']
        coordinates, pp, grid = self._get_parameters(kwargs, 'smeared_bao')

        ypred = emulator['model'](pp.reshape(-1,9), training=False)
        pk_bao = np.squeeze(np.exp(emulator['scaler'].inverse_transform(ypred)))

        if k is not None:
            if (max(k) > max(emulator['k']))|(min(k) < min(emulator['k'])):
                raise ValueError(f"""
                    A minimum k > 0.001 h/Mpc and a maximum k < 30 h/Mpc
                    are required for the smeared-BAO emulator:
                    the current values are {min(k)} h/Mpc and {max(k)} h/Mpc
                    """)
            else:
                pk_bao_interp = interpolate.interp1d(np.log(emulator['k']), np.log(pk_bao),
                kind='linear', axis = 0 if grid is None else 1, fill_value='extrapolate',
                assume_sorted=True)
                pk_bao = np.exp(pk_bao_interp(np.log(k)))
        else:
            k = emulator['k']
        return  k, pk_bao

    def get_no_wiggles_pk(self, omega_cold=None, omega_matter=None, omega_baryon=None,
                           sigma8_cold=None, A_s=None, hubble=None, ns=None, neutrino_mass=None,
                           w0=None, wa=None, expfactor=None, k=None, k_lin=None, pk_lin=None, **kwargs):
        """Evaluate the cold matter power spectrum with no wiggles (no bao), calling an emulator
            at a set of coordinates in parameter space.

        :param omega_cold: omega cold matter (cdm + baryons), either omega_cold
                           or omega_matter should be specified, if both are specified
                           they should be consistent
        :type omega_cold: float or array
        :param omega_matter: omega total matter (cdm + baryons + neutrinos), either omega_cold
                           or omega_matter should be specified, if both are specified
                           they should be consistent
        :type omega_matter: float or array
        :param sigma8_cold: rms of cold (cdm + baryons) linear perturbations, either sigma8_cold
                            or A_s should be specified, if both are specified they should be
                            consistent
        :type sigma8_cold: float or array
        :param A_s: primordial scalar amplitude at k=0.05 1/Mpc, either sigma8_cold
                    or A_s should be specified, if both are specified they should be
                    consistent
        :type A_s: float or array
        :param hubble: adimensional Hubble parameters, h=H0/(100 km/s/Mpc)
        :type hubble: float or array
        :param ns: scalar spectral index
        :type ns: float or array
        :param neutrino_mass: total neutrino mass
        :type neutrino_mass: float or array
        :param w0: dark energy equation of state redshift 0 parameter
        :type w0: float or array
        :param wa: dark energy equation of state redshift dependent parameter
        :type wa: float or array
        :param expfactor: expansion factor a = 1 / (1 + z)
        :type expfactor: float or array
        :param k: a vector of wavemodes in h/Mpc at which the nonlinear boost will be computed, if None
                the default wavemodes of the linear emulator will be used, defaults to None
        :type k: array_like, optional
        :param k_lin: a vector of wavemodes in h/Mpc, if None the wavemodes used by
                  the linear emulator are returned, defaults to None
        :type k_lin: array_like, optional
        :param pk_lin: a vector of linear power spectrum computed at k_lin, if None
                  the linear emulator will be called, defaults to None
        :type pk_lin: array_like, optional
        :param grid: dictionary with parameters and vectors of values where to evaluate the emulator, defaults to None
        :type grid: array_like, optional

        :return: k and the linear P(k)
        :rtype: tuple
        """

        _kwargs = locals()
        kwargs = {key: _kwargs[key] for key in set(list(_kwargs.keys())) - set(['self'])}

        if not self.compute_no_wiggles:
            raise ValueError("Please enable the no_wiggles emulator!")

        emulator = self.emulator['no_wiggles']
        coordinates, pp, grid = self._get_parameters(kwargs, 'no_wiggles')

        ypred = emulator['model'](pp.reshape(-1,7), training=False)
        ypred = emulator['scaler'].inverse_transform(ypred)
        ypred[:,emulator['k']>0.8] = 0.
        ypred = np.squeeze(ypred)

        if k is not None:
            if (min(k) < min(emulator['k'])) and (self.verbose):
                print(f"""The no-wiggles emulator is extrapolating from {min(k)} to {min(emulator['k'])} h/Mpc!""")

            pk_no_wiggles_interp = interpolate.interp1d(np.log(emulator['k']), ypred,
            kind='linear', axis = 0 if grid is None else 1, fill_value=0., bounds_error=False,
            assume_sorted=True)
            pk_no_wiggles_plin_ratio = np.exp(pk_no_wiggles_interp(np.log(k)))
        else:
            k = emulator['k']
            pk_no_wiggles_plin_ratio = np.squeeze(np.exp(ypred))

        if (k_lin is not None) & (pk_lin is not None):
            do_interp = False
            if len(k_lin) != len(k):
                do_interp = True
            else:
                if np.any(k_lin != k):
                    do_interp = True
            if do_interp:
                pk_lin_interp = interpolate.interp1d(np.log(k_lin), np.log(pk_lin),
                kind='linear', axis = 0 if grid is None else 1, fill_value='extrapolate',
                assume_sorted=True)
                pk_lin = np.exp(pk_lin_interp(np.log(k)))

        else:
            linear_kwargs = copy.deepcopy(kwargs)
            linear_kwargs['k'] = k
            k_lin, pk_lin = self.get_linear_pk(**linear_kwargs)

        pk_no_wiggles = pk_no_wiggles_plin_ratio * pk_lin

        return  k, pk_no_wiggles

    def get_sigma8(self, cold=True, omega_cold=None, omega_matter=None, omega_baryon=None,
                   sigma8_cold=None, A_s=None, hubble=None, ns=None, neutrino_mass=None,
                   w0=None, wa=None, expfactor=None, **kwargs):
        """Return sigma8, calling an emulator
            at a set of coordinates in parameter space.

        :param cold: whether to return sigma8_cold (cdm+baryons) or total matter
        :type cold: bool
        :param omega_cold: omega cold matter (cdm + baryons), either omega_cold
                           or omega_matter should be specified, if both are specified
                           they should be consistent
        :type omega_cold: float or array
        :param omega_matter: omega total matter (cdm + baryons + neutrinos), either omega_cold
                           or omega_matter should be specified, if both are specified
                           they should be consistent
        :type omega_matter: float or array
        :param sigma8_cold: rms of cold (cdm + baryons) linear perturbations, either sigma8_cold
                            or A_s should be specified, if both are specified they should be
                            consistent
        :type sigma8_cold: float or array
        :param A_s: primordial scalar amplitude at k=0.05 1/Mpc, either sigma8_cold
                    or A_s should be specified, if both are specified they should be
                    consistent
        :type A_s: float or array
        :param hubble: adimensional Hubble parameters, h=H0/(100 km/s/Mpc)
        :type hubble: float or array
        :param ns: scalar spectral index
        :type ns: float or array
        :param neutrino_mass: total neutrino mass
        :type neutrino_mass: float or array
        :param w0: dark energy equation of state redshift 0 parameter
        :type w0: float or array
        :param wa: dark energy equation of state redshift dependent parameter
        :type wa: float or array
        :param expfactor: expansion factor a = 1 / (1 + z)
        :type expfactor: float or array

        :return: sigma8
        :rtype: float or array
        """
        _kwargs = locals()
        kwargs = {key: _kwargs[key] for key in set(list(_kwargs.keys())) - set(['self'])}

        if self.compute_sigma8:
            emulator = self.emulator['sigma8']
        else:
            raise ValueError("Please enable the sigma8 emulator!")

        coordinates, pp, grid = self._get_parameters(kwargs, 'sigma8')
        ypred = np.squeeze(emulator['model'](pp.reshape(-1,7), training=False))
        if hasattr(coordinates['A_s'], '__len__'):
            ypred = (ypred.T * np.sqrt(coordinates['A_s']/1.e-9)).T
        else:
            ypred = ypred * np.sqrt(coordinates['A_s']/1.e-9)
        s8_index = 0 if cold else 1
        res = ypred[...,s8_index]
        res = float(res) if res.ndim==0 else res
        return  res


    def get_sigma12(self, cold=True, omega_cold=None, omega_matter=None, omega_baryon=None,
                   sigma8_cold=None, A_s=None, hubble=None, ns=None, neutrino_mass=None,
                   w0=None, wa=None, expfactor=None, **kwargs):
        """Return sigma12, calling an emulator
            at a set of coordinates in parameter space.

        :param cold: whether to return sigma8_cold (cdm+baryons) or total matter
        :type cold: bool
        :param omega_cold: omega cold matter (cdm + baryons), either omega_cold
                           or omega_matter should be specified, if both are specified
                           they should be consistent
        :type omega_cold: float or array
        :param omega_matter: omega total matter (cdm + baryons + neutrinos), either omega_cold
                           or omega_matter should be specified, if both are specified
                           they should be consistent
        :type omega_matter: float or array
        :param sigma8_cold: rms of cold (cdm + baryons) linear perturbations, either sigma8_cold
                            or A_s should be specified, if both are specified they should be
                            consistent
        :type sigma8_cold: float or array
        :param A_s: primordial scalar amplitude at k=0.05 1/Mpc, either sigma8_cold
                    or A_s should be specified, if both are specified they should be
                    consistent
        :type A_s: float or array
        :param hubble: adimensional Hubble parameters, h=H0/(100 km/s/Mpc)
        :type hubble: float or array
        :param ns: scalar spectral index
        :type ns: float or array
        :param neutrino_mass: total neutrino mass
        :type neutrino_mass: float or array
        :param w0: dark energy equation of state redshift 0 parameter
        :type w0: float or array
        :param wa: dark energy equation of state redshift dependent parameter
        :type wa: float or array
        :param expfactor: expansion factor a = 1 / (1 + z)
        :type expfactor: float or array

        :return: sigma12
        :rtype: float or array
        """
        _kwargs = locals()
        kwargs = {key: _kwargs[key] for key in set(list(_kwargs.keys())) - set(['self'])}

        if self.compute_sigma8:
            emulator = self.emulator['sigma8']
        else:
            raise ValueError("Please enable the sigma8 emulator!")

        coordinates, pp, grid = self._get_parameters(kwargs, 'sigma8')
        ypred = np.squeeze(emulator['model'](pp.reshape(-1,7), training=False))
        ypred = ypred * np.sqrt(coordinates['A_s']/1.e-9)
        s8_index = 2 if cold else 3
        res = ypred[...,s8_index]
        res = float(res) if res.ndim==0 else res
        return  res


    def get_A_s(self, omega_cold=None, omega_matter=None, omega_baryon=None,
                sigma8_cold=None, A_s=None, hubble=None, ns=None, neutrino_mass=None,
                w0=None, wa=None, expfactor=None, **kwargs):
        """Return A_s, corresponding to a given sigma8_cold
            at a set of coordinates in parameter space.

        :param omega_cold: omega cold matter (cdm + baryons), either omega_cold
                           or omega_matter should be specified, if both are specified
                           they should be consistent
        :type omega_cold: float or array
        :param omega_matter: omega total matter (cdm + baryons + neutrinos), either omega_cold
                           or omega_matter should be specified, if both are specified
                           they should be consistent
        :type omega_matter: float or array
        :param sigma8_cold: rms of cold (cdm + baryons) linear perturbations, either sigma8_cold
                            or A_s should be specified, if both are specified they should be
                            consistent
        :type sigma8_cold: float or array
        :param A_s: primordial scalar amplitude at k=0.05 1/Mpc, either sigma8_cold
                    or A_s should be specified, if both are specified they should be
                    consistent
        :type A_s: float or array
        :param hubble: adimensional Hubble parameters, h=H0/(100 km/s/Mpc)
        :type hubble: float or array
        :param ns: scalar spectral index
        :type ns: float or array
        :param neutrino_mass: total neutrino mass
        :type neutrino_mass: float or array
        :param w0: dark energy equation of state redshift 0 parameter
        :type w0: float or array
        :param wa: dark energy equation of state redshift dependent parameter
        :type wa: float or array
        :param expfactor: expansion factor a = 1 / (1 + z)
        :type expfactor: float or array

        :return: A_s
        :rtype: float or array
        """
        _kwargs = locals()
        kwargs = {key: _kwargs[key] for key in set(list(_kwargs.keys())) - set(['self'])}

        coordinates, pp, grid = self._get_parameters(kwargs, 'linear')

        res = np.squeeze(coordinates['A_s'])

        return float(res) if res.ndim == 0 else res


def load_linear_emu(verbose=True):
    """Loads in memory the linear emulator described in Aricò et al. 2021.

    :return: a dictionary containing the emulator object
    :rtype: dict
    """

    if verbose:
        print('Loading linear emulator...')

    basefold = os.path.dirname(os.path.abspath(__file__))
    emulator_cold_name = (basefold + '/' +
                     "cold_matter_linear_emu_1.0.0")
    emulator_tot_name = (basefold + '/' +
                     "total_matter_linear_emu_1.0.0")

    old_names = [(basefold + '/' + "linear_emulator")]
    for old_name in old_names:
        if os.path.exists(old_name):
            import shutil
            shutil.rmtree(old_name)

    if (not os.path.exists(emulator_cold_name)):
        import urllib.request
        import tarfile
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        print('Downloading emulator data (2 Mb)...')
        urllib.request.urlretrieve(
            'https://bacco.dipc.org/cold_matter_linear_emu_1.0.0.tar',
            emulator_cold_name + '.tar',
            MyProgressBar())
        tf = tarfile.open(emulator_cold_name+'.tar', 'r')
        tf.extractall(path=basefold)
        tf.close()
        os.remove(emulator_cold_name + '.tar')

    if (not os.path.exists(emulator_tot_name)):
        import urllib.request
        import tarfile
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        print('Downloading emulator data (2 Mb)...')
        urllib.request.urlretrieve(
            'https://bacco.dipc.org/total_matter_linear_emu_1.0.0.tar',
            emulator_tot_name + '.tar',
            MyProgressBar())
        tf = tarfile.open(emulator_tot_name+'.tar', 'r')
        tf.extractall(path=basefold)
        tf.close()
        os.remove(emulator_tot_name + '.tar')

    customs = {
                "accuracy_exp_002": accuracy_exp_002,
                "accuracy_exp_005": accuracy_exp_005,
                "mean_absolute_exp_percentage_error":mean_absolute_exp_percentage_error
                }
    metrics_list = ["accuracy",accuracy_exp_002, accuracy_exp_005]

    emulator = {}
    emulator['emu_type'] = 'nn'
    emulator['model_cold'] = load_model(emulator_cold_name, custom_objects=customs, compile=False)
    emulator['model_cold'].compile(optimizer='adam', loss=mean_absolute_exp_percentage_error, metrics=metrics_list)

    file_to_read = open(f"{emulator_cold_name}/details.pickle", "rb")
    nn_cold_details = pickle.load(file_to_read)

    emulator['k'] = nn_cold_details['kk']
    emulator['scaler_cold'] = nn_cold_details['scaler']

    emulator['model_tot'] = load_model(emulator_tot_name, custom_objects=customs, compile=False)
    emulator['model_tot'].compile(optimizer='adam', loss=mean_absolute_exp_percentage_error, metrics=metrics_list)

    file_to_read = open(f"{emulator_tot_name}/details.pickle", "rb")
    nn_tot_details = pickle.load(file_to_read)

    emulator['scaler_tot'] = nn_tot_details['scaler']
    emulator['keys'] = ['omega_cold', 'omega_baryon',
                                'hubble', 'neutrino_mass', 'w0', 'wa', 'expfactor']
    emulator['full_keys'] = ['omega_cold', 'omega_baryon', 'A_s', 'ns',
                                'hubble', 'neutrino_mass', 'w0', 'wa', 'expfactor']

    emulator['bounds'] = nn_tot_details['bounds']#{key: nn_cold_details['bounds'][i] for i, key in enumerate(emulator['keys'])}

    if verbose:
        print('Linear emulator loaded in memory.')

    return emulator

def load_sigma8_emu(verbose=True):
    """Loads in memory an emulator to quickly pass from A_s to sigma8.

    :return: a dictionary containing the emulator object
    :rtype: dict
    """

    if verbose:
        print('Loading sigma8 emulator...')

    basefold = os.path.dirname(os.path.abspath(__file__))

    old_names = [(basefold + '/' + "sigma8_emu_1.0.0"),]
    for old_name in old_names:
        if os.path.exists(old_name):
            import shutil
            shutil.rmtree(old_name)

    emulator_name = (basefold + '/' +
                     "sigma8_emu_2.0.0")

    if (not os.path.exists(emulator_name)):
        import urllib.request
        import tarfile
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        print('Downloading emulator data (141 Kb)...')
        urllib.request.urlretrieve(
            'https://bacco.dipc.org/sigma8_emu_2.0.0.tar',
            emulator_name + '.tar',
            MyProgressBar())
        tf = tarfile.open(emulator_name+'.tar', 'r')
        tf.extractall(path=basefold)
        tf.close()
        os.remove(emulator_name + '.tar')


    customs = {
                "accuracy_exp_002": accuracy_exp_002,
                "accuracy_exp_005": accuracy_exp_005,
                "mean_absolute_exp_percentage_error":mean_absolute_exp_percentage_error
                }
    metrics_list = ["accuracy",accuracy_exp_002, accuracy_exp_005]

    emulator = {}
    emulator['emu_type'] = 'nn'
    emulator['model'] = load_model(emulator_name, custom_objects=customs, compile=False)
    emulator['model'].compile(optimizer='adam', loss=mean_absolute_exp_percentage_error, metrics=metrics_list)

    file_to_read = open(f"{emulator_name}/details.pickle", "rb")
    nn_details = pickle.load(file_to_read)
    emulator['bounds'] = nn_details['bounds']
    emulator['keys'] = ['omega_cold', 'omega_baryon', 'ns', 'hubble', 'neutrino_mass', 'w0', 'wa']

    if verbose:
        print('Sigma8 emulator loaded in memory.')

    return emulator

def load_smeared_bao_emu(verbose=True):
    """Loads in memory the smeared bao emulator.

    :return: a dictionary containing the emulator object
    :rtype: dict
    """

    if verbose:
        print('Loading smeared bao emulator...')

    basefold = os.path.dirname(os.path.abspath(__file__))

    old_names = [(basefold + '/' + "smeared_bao_emu"),
                    (basefold + '/' + "smeared_bao_emu_1")]
    for old_name in old_names:
        if os.path.exists(old_name):
            import shutil
            shutil.rmtree(old_name)

    emulator_name = (basefold + '/' +
                     "smeared_bao_emu_1.0.0")

    if (not os.path.exists(emulator_name)):
        import urllib.request
        import tarfile
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        print('Downloading emulator data (1.2 Mb)...')
        urllib.request.urlretrieve(
            'https://bacco.dipc.org/smeared_bao_emu_1.0.0.tar',
            emulator_name + '.tar',
            MyProgressBar())
        tf = tarfile.open(emulator_name+'.tar', 'r')
        tf.extractall(path=basefold)
        tf.close()
        os.remove(emulator_name + '.tar')

    customs = {
                "accuracy_exp_002": accuracy_exp_002,
                "accuracy_exp_005": accuracy_exp_005,
                "mean_absolute_exp_percentage_error":mean_absolute_exp_percentage_error
                }
    metrics_list = ["accuracy",accuracy_exp_002, accuracy_exp_005]

    emulator = {}
    emulator['emu_type'] = 'nn'
    emulator['model'] = load_model(emulator_name, custom_objects=customs, compile=False)
    emulator['model'].compile(optimizer='adam', loss=mean_absolute_exp_percentage_error, metrics=metrics_list)

    file_to_read = open(f"{emulator_name}/details.pickle", "rb")
    nn_details = pickle.load(file_to_read)
    emulator['k'] = nn_details['kk']
    emulator['scaler'] = nn_details['scaler']
    emulator['bounds'] = nn_details['bounds']
    emulator['keys'] = ['omega_cold', 'sigma8_cold', 'omega_baryon', 'ns',
                                        'hubble', 'neutrino_mass', 'w0', 'wa', 'expfactor']

    if verbose:
        print('Smeared bao emulator loaded in memory.')

    return emulator

def load_no_wiggles_emu(verbose=True):
    """Loads in memory the no-wiggles emulator.

    :return: a dictionary containing the emulator object
    :rtype: dict
    """

    if verbose:
        print('Loading no-wiggles emulator...')
    import deepdish as dd

    basefold = os.path.dirname(os.path.abspath(__file__))

    emulator_name = (basefold + '/' +
                     "no_wiggles_emu_1.0.0")

    if (not os.path.exists(emulator_name)):
        import urllib.request
        import tarfile
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        print('Downloading emulator data (6.7 Mb)...')
        urllib.request.urlretrieve(
            'https://bacco.dipc.org/no_wiggles_emu_1.0.0.tar',
            emulator_name + '.tar',
            MyProgressBar())
        tf = tarfile.open(emulator_name+'.tar', 'r')
        tf.extractall(path=basefold)
        tf.close()
        os.remove(emulator_name + '.tar')

    customs = {
                "accuracy_exp_002": accuracy_exp_002,
                "accuracy_exp_005": accuracy_exp_005,
                "mean_absolute_exp_percentage_error":mean_absolute_exp_percentage_error
                }
    metrics_list = ["accuracy",accuracy_exp_002, accuracy_exp_005]

    emulator = {}
    emulator['emu_type'] = 'nn'
    emulator['model'] = load_model(emulator_name, custom_objects=customs, compile=False)
    emulator['model'].compile(optimizer='adam', loss=mean_absolute_exp_percentage_error, metrics=metrics_list)

    file_to_read = open(f"{emulator_name}/details.pickle", "rb")
    nn_details = pickle.load(file_to_read)

    emulator['k'] = nn_details['kk']
    emulator['scaler'] = nn_details['scaler']
    emulator['bounds'] = nn_details['bounds']
    emulator['keys'] = nn_details['keys']

    if verbose:
        print('No-wiggles emulator loaded in memory.')

    return emulator

def load_nonlinear_emu(verbose=True, fold_name=None, detail_name=None):
    """Loads in memory the nonlinear emulator described in Angulo et al. 2020.

    :return: a dictionary containing the emulator object
    :rtype: dict
    """
    if verbose:
        print('Loading non-linear emulator...')

    if fold_name is None:
        basefold = os.path.dirname(os.path.abspath(__file__))

        old_emulator_names = [(basefold + '/' +
                            "NN_emulator_data_iter4_big_160.pickle_sg_0.95_2000_rot_bao"),
                            (basefold + '/' +
                            "NN_emulator_data_iter4_big_160.pickle_sg_0.99_2000_PCA5_BNFalse_DO0rot_bao"),
                            (basefold + '/' +
                            "NN_emulator_data_iter4_big_160.pickle_sg_0.99_2000_PCA6_BNFalse_DO0rot_bao")]
        for old_emulator_name in old_emulator_names:
            if os.path.exists(old_emulator_name):
                import shutil
                shutil.rmtree(old_emulator_name)

        emulator_name = (basefold + '/' +
                        "nonlinear_emu_1.0.1")
        detail_name = 'details.pickle'

        if (not os.path.exists(emulator_name)):
            import urllib.request
            import tarfile
            import ssl
            ssl._create_default_https_context = ssl._create_unverified_context
            print('Downloading emulator data (3Mb)...')
            urllib.request.urlretrieve(
                'https://bacco.dipc.org/nonlinear_emu_1.0.1.tar',
                emulator_name + '.tar',
                MyProgressBar())
            tf = tarfile.open(emulator_name+'.tar', 'r')
            tf.extractall(path=basefold)
            tf.close()
            os.remove(emulator_name + '.tar')
    else:
        emulator_name = fold_name
    emulator = {}
    emulator['emu_type'] = 'nn'
    emulator['model'] = load_model(emulator_name, compile=False)
    with open(os.path.join(emulator_name, detail_name), 'rb') as f:
        emulator['scaler'] = pickle.load(f)
        _pca = pickle.load(f)
        emulator['k'] = pickle.load(f)
        _trcoords = pickle.load(f)
        emulator['bounds'] = pickle.load(f)

    emulator['keys'] = ['omega_cold', 'sigma8_cold', 'omega_baryon', 'ns',
                                'hubble', 'neutrino_mass', 'w0', 'wa', 'expfactor']
    if verbose:
        print('Nonlinear emulator loaded in memory.')
    return emulator

def _compute_camb_spectrum(params, kmax=50, k_per_logint=0, cold=True):
    """
    Calls camb with the current cosmological parameters and returns a
    dictionary with the following keys:
    kk, pk
    :param cold: whether to return the cold matter power spectrum or the total one. Default to cold.
    :type cold: bool, optional

    Through the species keyword the following power spectra can be obtained:
    matter, cdm, baryons, neutrinos, cold matter (cdm+baryons), photons,
    divergence of the cdm velocity field, divergence of the baryon velocity
    field, divergence of the cdm-baryon relative velocity field
    """
    import camb

    if 'tau' not in params.keys():
        params['tau'] = 0.0952
    if 'num_massive_neutrinos' not in params.keys():
        params['num_massive_neutrinos'] = 3 if params['neutrino_mass'] != 0.0 else 0
    if 'Neffective' not in params.keys():
        params['Neffective'] = 3.046
    if 'omega_k' not in params.keys():
        params['omega_k'] = 0
    if 'omega_cdm' not in params.keys():
        if 'omega_cold' in params.keys():
            params['omega_cdm'] = params['omega_cold'] - params['omega_baryon']
        elif 'omega_matter' in params.keys():
            params['omega_cdm'] = params['omega_matter'] - params['omega_baryon'] - params['neutrino_mass'] / 93.14 / params['hubble']**2
        else:
            raise ValueError('At least one among omega_matter and omega_cold should be specified')

    assert params['omega_k'] == 0, 'Non flat geometries are not supported'

    expfactor = params['expfactor']

    # Set up a new set of parameters for CAMB
    pars = camb.CAMBparams()

    # This function sets up CosmoMC-like settings, with one massive neutrino and helium set using BBN consistency
    # Set neutrino-related parameters
    # camb.nonlinear.Halofit('takahashi')
    pars.set_cosmology(
        H0=100 * params['hubble'],
        ombh2=(params['omega_baryon'] * params['hubble']**2),
        omch2=(params['omega_cdm'] * params['hubble']**2),
        omk=params['omega_k'],
        neutrino_hierarchy='degenerate',
        num_massive_neutrinos=params['num_massive_neutrinos'],
        mnu=params['neutrino_mass'],
        standard_neutrino_neff=params['Neffective'],
        tau=params['tau'])


    if 'A_s' in params.keys():
        if params['A_s'] is not None:
            A_s = params['A_s']
            ReNormalizeInputSpectrum = False
        else:
            A_s = 2.e-9
            ReNormalizeInputSpectrum = True
    else:
        A_s = 2.e-9
        ReNormalizeInputSpectrum = True

    pars.set_dark_energy(
        w=params['w0'],
        wa=params['wa'])

    redshifts = [(1 / expfactor - 1)]
    if expfactor < 1:
        redshifts.append(0)

    pars.InitPower.set_params(ns=params['ns'], As=A_s)
    pars.YHe = 0.24
    pars.set_matter_power(
        redshifts=redshifts,
        kmax=kmax,
        k_per_logint=k_per_logint)

    pars.WantCls = False
    pars.WantScalars = False
    pars.Want_CMB = False
    pars.DoLensing = False

    # calculate results for these parameters
    results = camb.get_results(pars)

    if cold:
        index = 7 # cdm + baryons
    else:
        index = 6
    kh, z, pk = results.get_linear_matter_power_spectrum(var1=(1 + index),
                                                         var2=(1 + index),
                                                         hubble_units=True,
                                                         have_power_spectra=False,
                                                         params=None)
    pk = pk[-1, :]

    if ReNormalizeInputSpectrum:
        sigma8 = results.get_sigmaR(8, z_indices=-1, var1=(1 + index), var2=(1 + index))
        if cold:
            Normalization = (params['sigma8_cold'] / sigma8)**2
        else:
            Normalization = (params['sigma8'] / sigma8)**2
        pk *= Normalization

    mask = (kh > 1e-4)

    return {'k': kh[mask], 'pk': pk[mask]}

def compute_camb_pk(coordinates, k=None, cold=True):
    """Compute the linear prediction of the matter power spectrum using camb

    The coordinates must be specified as a dictionary with the following
    keywords:

    #. 'omega_cold'
    #. 'omega_baryon'
    #. 'sigma8'
    #. 'hubble'
    #. 'ns'
    #. 'neutrino_mass'
    #. 'w0'
    #. 'wa'
    #. 'expfactor'

    :param coordinates: a set of coordinates in parameter space
    :type coordinates: dict
    :param k: a vector of wavemodes in h/Mpc, if None the wavemodes used by
              camb are returned, defaults to None
    :type k: array_like, optional
    :param cold: whether to return the cold matter power spectrum or the total one. Default to cold.
    :type cold: bool, optional
    :return: k and linear pk
    :rtype: tuple
    """
    _pk_dict = _compute_camb_spectrum(coordinates, cold=cold)
    if k is not None:
        _k = k
        _interp = interpolate.interp1d(np.log(_pk_dict['k']), np.log(_pk_dict['pk']), kind='cubic')
        _pk = np.exp(_interp(np.log(_k)))
    else:
        _k = _pk_dict['k']
        _pk = _pk_dict['pk']
    return _k, _pk

def _nowiggles_pk(k_lin=None, pk_lin=None, k_emu=None):
    """De-wiggled linear prediction of the cold matter power spectrum

    The BAO feature is removed by identifying and removing its corresponding
    bump in real space, by means of a DST, and consequently transforming
    back to Fourier space.
    See:
    - Baumann et al 2018 (https://arxiv.org/pdf/1712.08067.pdf)
    - Giblin et al 2019 (https://arxiv.org/pdf/1906.02742.pdf)

    :param k_lin: a vector of wavemodes in h/Mpc, if None the wavemodes used by
              camb are returned, defaults to None
    :type k_lin: array_like, optional
    :param pk_lin: a vector of linear power spectrum computed at k_lin, if None
              camb will be called, defaults to None
    :type pk_lin: array_like, optional

    :param k_emu: a vector of wavemodes in h/Mpc, if None the wavemodes used by
              the emulator are returned, defaults to None
    :type k_emu: array_like, optional

    :return: dewiggled pk computed at k_emu
    :rtype: array_like
    """

    from scipy.fftpack import dst, idst

    nk = int(2**15)
    kmin = k_lin.min()
    kmax = 10
    klin = np.linspace(kmin, kmax, nk)

    pkcamb_cs = interpolate.splrep(np.log(k_lin), np.log(pk_lin), s=0)
    pklin = np.exp(interpolate.splev(np.log(klin), pkcamb_cs, der=0, ext=0))

    f = np.log10(klin * pklin)

    dstpk = dst(f, type=2)

    even = dstpk[0::2]
    odd = dstpk[1::2]

    i_even = np.arange(len(even)).astype(int)
    i_odd = np.arange(len(odd)).astype(int)

    even_cs = interpolate.splrep(i_even, even, s=0)
    odd_cs = interpolate.splrep(i_odd, odd, s=0)

    even_2nd_der = interpolate.splev(i_even, even_cs, der=2, ext=0)
    odd_2nd_der = interpolate.splev(i_odd, odd_cs, der=2, ext=0)

    # these indexes have been fudged for the k-range considered
    # [~1e-4, 10], any other choice would require visual inspection
    imin_even = i_even[100:300][np.argmax(even_2nd_der[100:300])] - 20
    imax_even = i_even[100:300][np.argmin(even_2nd_der[100:300])] + 70
    imin_odd = i_odd[100:300][np.argmax(odd_2nd_der[100:300])] - 20
    imax_odd = i_odd[100:300][np.argmin(odd_2nd_der[100:300])] + 75

    i_even_holed = np.concatenate((i_even[:imin_even], i_even[imax_even:]))
    i_odd_holed = np.concatenate((i_odd[:imin_odd], i_odd[imax_odd:]))

    even_holed = np.concatenate((even[:imin_even], even[imax_even:]))
    odd_holed = np.concatenate((odd[:imin_odd], odd[imax_odd:]))

    even_holed_cs = interpolate.splrep(i_even_holed, even_holed * (i_even_holed+1)**2, s=0)
    odd_holed_cs = interpolate.splrep(i_odd_holed, odd_holed * (i_odd_holed+1)**2, s=0)

    even_smooth = interpolate.splev(i_even, even_holed_cs, der=0, ext=0) / (i_even + 1)**2
    odd_smooth = interpolate.splev(i_odd, odd_holed_cs, der=0, ext=0) / (i_odd + 1)**2

    dstpk_smooth = []
    for ii in range(len(i_even)):
        dstpk_smooth.append(even_smooth[ii])
        dstpk_smooth.append(odd_smooth[ii])
    dstpk_smooth = np.array(dstpk_smooth)

    pksmooth = idst(dstpk_smooth, type=2) / (2 * len(dstpk_smooth))
    pksmooth = 10**(pksmooth) / klin

    k_highk = k_lin[k_lin > 5]
    p_highk = pk_lin[k_lin > 5]

    k_extended = np.concatenate((klin[klin < 5], k_highk))
    p_extended = np.concatenate((pksmooth[klin < 5], p_highk))

    pksmooth_cs = interpolate.splrep(np.log(k_extended), np.log(p_extended), s=0)
    pksmooth_interp = np.exp(interpolate.splev(np.log(k_emu), pksmooth_cs, der=0, ext=0))

    return pksmooth_interp

def _smeared_bao_pk(k_lin=None, pk_lin=None, k_emu=None, pk_lin_emu=None, pk_nw=None, grid=None):
    """Prediction of the cold matter power spectrum using a Boltzmann solver with smeared BAO feature

    :param k_lin: a vector of wavemodes in h/Mpc, if None the wavemodes used by
              camb are returned, defaults to None
    :type k_lin: array_like, optional
    :param pk_lin: a vector of linear power spectrum computed at k_lin, if None
              camb will be called, defaults to None
    :type pk_lin: array_like, optional

    :param k_emu: a vector of wavemodes in h/Mpc, if None the wavemodes used by
              the emulator are returned, defaults to None
    :type k_emu: array_like, optional
    :param pk_emu: a vector of linear power spectrum computed at k_emu, defaults to None
    :type pk_emu: array_like, optional
    :param pk_nw: a vector of no-wiggles power spectrum computed at k_emu, defaults to None
    :type pk_nw: array_like, optional
    :param grid: dictionary with parameter and vector of values where to evaluate the emulator, defaults to None
    :type grid: array_like, optional

    :return: smeared BAO pk computed at k_emu
    :rtype: array_like
    """
    try:
        from scipy.integrate import trapezoid
    except ImportError:
        from scipy.integrate import trapz as trapezoid

    if grid is None:
        sigma_star_2 = trapezoid(k_lin * pk_lin, x=np.log(k_lin)) / (3 * np.pi**2)
        k_star_2 = 1 / sigma_star_2
        G = np.exp(-0.5 * (k_emu**2 / k_star_2))
        if pk_nw is None:
            pk_nw = _nowiggles_pk(k_lin=k_lin, pk_lin=pk_lin, k_emu=k_emu)
    else:
        sigma_star_2 = trapezoid(k_lin[None,:] * pk_lin, x=np.log(k_lin[None:,]), axis=1) / (3 * np.pi**2)
        k_star_2 = 1 / sigma_star_2
        G = np.exp(-0.5 * (k_emu**2 / k_star_2[:,None]))
        if pk_nw is None:
            pk_nw = np.array([_nowiggles_pk(k_lin=k_lin, pk_lin=pkl, k_emu=k_emu) for pkl in pk_lin])
    return pk_lin_emu * G + pk_nw * (1 - G)
