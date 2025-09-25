"""
Author: Sunao Sugiyama
Last edit: 2023.05.17

This computes the magnification bias on g-g lensing in physical scale, i.e. dSigma.

Code reviewed by Tianqing Zhang in Oct 2023
"""
import numpy as np
from scipy import integrate
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from fftlog import hankel, fftlog

class magnification_class:
    def __init__(self, config = None):
        self.config = {'verbose':True}
        self.config.update(config or {})
    
    def set_param(self, alpha, Omm):
        """
        alpha (float): magnification bias parameter
        """
        self.param = {'alpha': alpha, 'Omm':Omm}
        
    def set_pk_nlin_data(self, z, k, pknonlin):
        """
        set nonlinear matter power spectrum
        """
        self.pk_nlin_data = [z, k, pknonlin]
        
    def set_z2chi(self, z, chi):
        """
        set relation between redshift z and comoving distance chi.
        """
        self.z2chi = ius(z, chi)
        
    def set_nz_source(self, z, nz):
        """
        Here, input redshfit distribution should already include redshift bias effect.
        """
        dtype = np.dtype([('z', z.dtype), ('nz', nz.dtype)])
        self.nzs = np.empty(z.size, dtype)
        self.nzs['z']  = z
        self.nzs['nz'] = nz
        
    def set_nz_lens(self, z, nz):
        dtype = np.dtype([('z', z.dtype), ('nz', nz.dtype)])
        self.nzl = np.empty(z.size, dtype)
        self.nzl['z']  = z
        self.nzl['nz'] = nz
    
    def _kzpktable2pltable(self, z, k, pktable, l):
        """
        Convert P(k_i, z_j) -> P(l_i/chi_j, z_j)
        """
        pltable = np.empty((l.size, z.size))
        chi = self.z2chi(z) # [Mpc/h]
        for i,_z in enumerate(z):
            pltable[:,i] = log_extrap_func(k, pktable[i, :])((l+0.5)/chi[i])
        return pltable
    
    def _get_clSigmacrit_mag(self):
                
        window = get_Sigmacr_Cl_window(self.nzs['z'], self.nzs['nz'], 
                                       self.nzl['z'], self.nzl['nz'], 
                                       self.z2chi) # [h/Mpc]
        H0 = 100/299792.4580 # [h/Mpc]
        Om= self.param['Omm']
        rhocrit = 2.77536627e11 # h^2 M_\odot/Mpc^3
        prefactor = (2.0/3.0*rhocrit/H0**2)*(3.0/2.0*H0**2*Om)**2 # [h^4Msun/Mpc^5]
        
        l = np.logspace(0, 5, 1000)
        sel = self.pk_nlin_data[0] > 0
        pltable = self._kzpktable2pltable(self.pk_nlin_data[0][sel], 
                                          self.pk_nlin_data[1], 
                                          self.pk_nlin_data[2][sel, :], l)
        
        chi= self.z2chi(self.pk_nlin_data[0][sel]) # [Mpc/h]
        integrand   = window(chi)*pltable
        clSigmacrit = integrate.simps(integrand.T, chi, axis=0)
        clSigmacrit*= prefactor/1e12 # hMsun/pc^2
        return l, clSigmacrit
    
    def get_ds_mag(self, z, rp, dlnrp, N_extrap_low=None, N_extrap_high=None):
        """
        rp     (array): array of radial bins in unit of Mpc/h, need to be corrected for the measurement effect.
        """
        l, clSigmacrit = self._get_clSigmacrit_mag()
        # print(self.nzs['z'])
        # print(clSigmacrit)
        if N_extrap_low is None:
            N_extrap_low = l.size
        if N_extrap_high is None:
            N_extrap_high = l.size
        if np.any(clSigmacrit < 0):
            raise "some of the clSigmacrit in magnification bias have negative (nonphysical) values."
        myhankel = hankel(l, l**2*clSigmacrit, 1.01, 
                          N_extrap_low=N_extrap_low,
                          N_extrap_high=N_extrap_high,
                          c_window_width=0.25)
    
        if dlnrp > 0:
            D = 2 # dimension
            t, ds_mag = myhankel.hankel_binave(2, dlnrp, D)
        else:
            t, ds_mag = myhankel.hankel(2)
        ds_mag /= 2.0*np.pi

        chil_rep = self.z2chi(z) # [Mpc/h]
        ds_mag = ius(t, ds_mag, ext=3)(rp/chil_rep)
        ds_mag*= 2*(self.param['alpha']-1)
        
        return ds_mag


def get_Sigmacr_Cl_window(zs_in, nzs_in, zl_in, nzl_in, z2chi):
    if isinstance(zl_in, (int, float)) and isinstance(nzl_in, (int, float)):
        zl = np.array([zl_in])
        nzl = np.array([1])
        dzl = 1
    else:
        # discard nz with 0
        sel = nzl_in > 0
        zl = zl_in[sel]
        nzl= nzl_in[sel]
        dzl = np.diff(zl)[0]
        assert np.all(np.isclose(np.diff(zl), dzl))
    if isinstance(zs_in, (int, float)) and isinstance(nzs_in, (int, float)):
        zs = np.array([zs])
        nzs = np.array([1])
        dzs = 1
    else:
        # discard nz with 0
        sel = nzs_in > 0
        zs = zs_in[sel]
        nzs= nzs_in[sel]
        dzs = np.diff(zs)[0]
        # print(np.diff(zs))
        # assert np.all(np.isclose(np.diff(zs), dzs, rtol=0.01)), "photoz bin must be linear within 1% "
    
    nzs_normed = nzs/(np.sum(nzs)*dzs)
    nzl_normed = nzl/(np.sum(nzl)*dzl)
    
    chil = z2chi(zl) # Mpc/h
    chis = z2chi(zs) # Mpc/h
    # print(chil)
    # print(chis)
    
    c0, c1, c2 = [], [], []
    for _zs, _nzs, _chis in zip(zs, nzs_normed, chis):
        _c0, _c1, _c2 = [], [], []
        for _zl, _nzl, _chil in zip(zl, nzl_normed, chil):
            if _zl >= _zs:
                _c0.append(0.0)
                _c1.append(0.0)
                _c2.append(0.0)
            else:
                if 1+_zl==0.0:
                    print("1+zl = ", 1+_zl)
                if _chil==0.0:
                    print("_chil= ", _chil)
                if np.abs(_chis-_chil)<1e-3:
                    # print("(_chis-_chil)= ", (_chis-_chil))
                    _c0.append(0.0)
                    _c1.append(0.0)
                    _c2.append(0.0)
                    continue
                _c0.append(_nzs*_nzl/(1+_zl)/_chil**2/(_chis-_chil) * _chil*_chis )
                _c1.append(_nzs*_nzl/(1+_zl)/_chil**2/(_chis-_chil) * (_chil+_chis) )
                _c2.append(_nzs*_nzl/(1+_zl)/_chil**2/(_chis-_chil) )
        c0.append(_c0)
        c1.append(_c1)
        c2.append(_c2)
        
    c0 = np.array(c0)
    c1 = np.array(c1)
    c2 = np.array(c2)
    
    # print(c0,c1,c2)
    
    zlMat, zsMat = np.meshgrid(zl, zs)
    assert np.all(c0.shape == zlMat.shape)
    assert np.all(c0.shape == zsMat.shape)
    
    z = np.linspace(0.0, zl[zl>0].max()*1.01, 100)
    chi = z2chi(z) # Mpc/h
    c0_chi, c1_chi, c2_chi = [], [], []
    for _z, _chi in zip(z, chi):
        mask = np.logical_and(zlMat>_z, zsMat>_z, zsMat>zlMat)
        c0_chi.append( np.sum(c0[mask])*dzl*dzs )
        c1_chi.append( np.sum(c1[mask])*dzl*dzs )
        c2_chi.append( np.sum(c2[mask])*dzl*dzs )
    c0_chi = np.array(c0_chi)
    c1_chi = np.array(c1_chi)
    c2_chi = np.array(c2_chi)
    # print(c0_chi, c1_chi, c2_chi)
    
    window = (c0_chi - c1_chi*chi + c2_chi*chi**2)*(1+z)**2

    window = ius(chi, window, ext=1) # [h/Mpc]
    return window
    
def log_extrap_func(x,y):
    def func(x_new):
        if isinstance(x_new, (int, float)):
            x_new = np.atleast_1d(x_new)
        ans = np.zeros(x_new.size)
        sel = x_new < x.min()
        if np.sum(sel):
            ans[sel] = np.exp( np.log(y[1]/y[0])/np.log(x[1]/x[0]) * np.log(x_new[sel]/x[0]) + np.log(y[0]) )
        sel = x.max() < x_new
        if np.sum(sel):
            ans[sel] = np.exp( np.log(y[-2]/y[-1])/np.log(x[-2]/x[-1]) * np.log(x_new[sel]/x[-1]) + np.log(y[-1]) )
        sel = np.logical_and(x.min()<= x_new, x_new <= x.max())
        ans[sel] = 10**ius(np.log10(x),np.log10(y))(np.log10(x_new[sel]))
        return ans
    return func
    