"""
Author: Sunao Sugiyama
Last edit: 2023.05.19

This is the measureemnt correction module for the projected correlations, dSigma and wp.
This is independent from the COSMOSIS or other cosmological likelihood packages, and thus
posrtable module.

Code reviewed by Tianqing Zhang in Oct 2023
"""
import numpy as np
import astropy.cosmology
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy.interpolate import RegularGridInterpolator
import time
from astropy.io import fits

from astropy import constants as const
c = const.c.to('Mpc/s').value
G = const.G.to('Mpc^3/s^2kg^1').value
M_sun = const.M_sun.to('g').value
sigcrit_prefactor = c**2/(4.*np.pi*G)*10**3/M_sun/10**12
del c, G, M_sun # delete to save name space

class dSigma_meascorr_class:
    def __init__(self, config):
        """
        This function computes the overall correction factor multiplied to dSigma signal on model side.
        
        Args:
          config (dict) : Following keys must be included
            - Omm   : Omega_m used for measurement
            - w0    : Equation of state of dark energy used for measurement
            - name  : name of this instance
            - bin0  : binid of lens
            - bin1  : binid of source
        
        https://arxiv.org/abs/2111.10966
        """
        self.config = config

    def set_zs(self, zs):
        self.zs = zs
        
    def set_zl(self, zl):
        self.zl = zl
        
    def set_sumwlssigcritinvPz(self, sumwlssigcritinvPz):
        self.sumwlssigcritinvPz = sumwlssigcritinvPz
        
    def _get_cosmo(self, Omm, w0):
        return astropy.cosmology.FlatwCDM(100, Omm, w0=w0)
    
    def _get_dSigma_corr_denominator(self, dpz, Omm, w0):
        zl = self.zl
        #zs = self.zs - dpz # This is the definition in HSC Y1 2x2pt and HSC Y3 3x2pt analyses.
        zs = self.zs + dpz # This is the definition of CosmoSIS
        sumwlssigcritinvPz = self.sumwlssigcritinvPz/np.sum(self.sumwlssigcritinvPz)
        cosmo = self._get_cosmo(Omm, w0)

        zl_rough = np.linspace(zl.min(), zl.max()+0.001,100) # Max has padding
        chi_zl = ius(zl_rough, cosmo.comoving_distance(zl_rough).value)(zl)
        chi_zs = cosmo.comoving_distance(zs).value

        dSigma_corr_numerator = 0.0
        for j in range(len(zs)):
            if chi_zs[j] <= 0.0:
                continue
            Sigma_cr_inv_array = 1./sigcrit_prefactor * (1.+zl)*chi_zl*(1.-chi_zl/chi_zs[j])
            Sigma_cr_inv_array[Sigma_cr_inv_array <0.0] = 0.0
            dSigma_corr_numerator += np.sum(Sigma_cr_inv_array*sumwlssigcritinvPz[:,j])

        return dSigma_corr_numerator
    
    def _get_dSigma_corr_numerator(self, dpz, Omm, w0, bin1, z, nz):
        zl = self.zl
        #zs = self.zs - dpz # This is the definition in HSC Y1 2x2pt and HSC Y3 3x2pt analyses.
        zs = z + dpz # This is the definition of CosmoSIS
        sumwlssigcritinvPz = nz/np.sum(nz)/len(zl)
        
        cosmo = self._get_cosmo(Omm, w0)

        zl_rough = np.linspace(zl.min(), zl.max()+0.001,100) # Max has padding
        chi_zl = ius(zl_rough, cosmo.comoving_distance(zl_rough).value)(zl)
        chi_zs = cosmo.comoving_distance(zs).value

        dSigma_corr_numerator = 0.0
        for j in range(len(zs)):
            if chi_zs[j] <= 0.0:
                continue
            Sigma_cr_inv_array = 1./sigcrit_prefactor * (1.+zl)*chi_zl*(1.-chi_zl/chi_zs[j])
            Sigma_cr_inv_array[Sigma_cr_inv_array <0.0] = 0.0
            dSigma_corr_numerator += np.sum(Sigma_cr_inv_array*sumwlssigcritinvPz[j])

        return dSigma_corr_numerator
    
    def get_corr_factor(self, dpz, Omm, w0, bin1, z, nz):
        # dSigma_corr = self._get_dSigma_corr_numerator(dpz, Omm, w0, bin1, z, nz)/self._get_dSigma_corr_denominator(0.0, self.config['Omm'], self.config['w0'])
        # print(dpz, Omm, w0)
        # print(0.0, self.config['Omm'], self.config['w0'])
        dSigma_corr = self._get_dSigma_corr_denominator(dpz, Omm, w0)/self._get_dSigma_corr_denominator(0.0, self.config['Omm'], self.config['w0'])
        
        # print(dSigma_corr)
        
        return dSigma_corr
    
    def reduce(self, nbin):
        """
        The original output by weaklens_pipeline has sumwlssigcritinvPz file with shape of 
        (# of lens galaxies, # of photo-z bin). This is huge indeed, and the computation of 
        this measurement correction can be the bottle neck for the likelihood evaluation.
        To avoid this, we reduce the size of the matrix by binning the lens redshift, i.e.
        binning the first axis.
        
        Args:
            nbin: number of bins for the lens redshift.
        """
        zmin = self.zl.min()
        zmax = self.zl.max()
        zlbin_lowedge = np.linspace(zmin, zmax, nbin+1)[:-1]
        delta_z = zlbin_lowedge[1]-zlbin_lowedge[0]
        zlbin = zlbin_lowedge + delta_z/2.0
        
        # indix of bin in which the individual lens redshift falls
        idx = np.array((self.zl-zmin)/delta_z, dtype=int) 
        # initialize the binned sumwlssigcritinvPz
        sumwlssigcritinvPz_lensbin_ave = np.empty((nbin, self.sumwlssigcritinvPz.shape[1]))
        for i in range(nbin):
            sel = idx == i
            sumwlssigcritinvPz_lensbin_ave[i,:] = np.sum(self.sumwlssigcritinvPz[sel, :], axis=0)
            
        # replace
        self.zl = zlbin
        self.sumwlssigcritinvPz = sumwlssigcritinvPz_lensbin_ave
        
    def stack(self, newone):
        assert np.all(self.zl == newone.zl), 'lens redshift does not match.'
        assert np.all(self.sumwlssigcritinvPz.shape == newone.sumwlssigcritinvPz.shape),\
                'shapes of sumwlssigcritinvPz mismatch'
        
        self.sumwlssigcritinvPz += newone.sumwlssigcritinvPz
        
    def load_weaklens_pipeline_output(self, fname):
        d = np.loadtxt(fname)
        # 0 th column is the index of lens galaxy
        self.set_zl(d[:, 1])
        self.set_sumwlssigcritinvPz(d[:, 2:])
        
    def load_photoz_bin(self, fname):
        d = np.loadtxt(fname)
        self.set_zs(d)
        
    def dump(self):
        data = np.vstack([np.arange(self.zl.size), self.zl, self.sumwlssigcritinvPz.T]).T
        return data
    
    def save(self, fname):
        np.savetxt(fname, self.dump())
        
    def to_fits(self):
        header = fits.Header()
        header['Omm'] = self.config['Omm']
        header['w0']  = self.config['w0']
        header['EXTNAME'] = self.config['name']
        header['BIN0'] = self.config['bin0']
        header['BIN1'] = self.config['bin1']
        header['SUMPZ'] = True
        header['INDEX'] = 0 # to indicate that the first column is the lens index
        header['LENSZ'] = 1 # to indicate that the second column is the lens redshift
        header['SMPZMAT'] = 2 # to indicate that the third and followings columns are the sumwlssigcritinvPz
        extension = fits.ImageHDU(data=self.dump(), header=header)
        return extension
    
    def get_bin_pair(self):
        return self.config['bin0'], self.config['bin1']
    
    @classmethod
    def from_fits(cls, ext):
        config = {'Omm': ext.header['Omm'], 
                  'w0': ext.header['w0'], 
                  'name': ext.header['EXTNAME'], 
                  'bin0': ext.header['BIN0'], 
                  'bin1': ext.header['BIN1']}
        ins = cls(config)
        ins.set_sumwlssigcritinvPz(ext.data[:, 2:])
        ins.set_zl(ext.data[:, 1])
        return ins
        
    @classmethod
    def load_reduce_dump(cls, nbin, fname_in, fname_out):
        ins = cls({})
        ins.load_weaklens_pipeline_output(fname_in)
        ins.reduce(nbin)
        ins.save(fname_out)
        
        
class wp_meascorr_class:
    def __init__(self, config):
        """
        This class deals with the measureemnt correction on the pimax of wp.
        Args:
          config (dict) : Following keys must be included
            - Omm   : Omega_m used for measurement
            - w0    : Equation of state of dark energy used for measurement
        
        https://arxiv.org/abs/2111.10966
        """
        self.config = config
        
    def _get_cosmo(self, Omm, w0):
        return astropy.cosmology.FlatwCDM(100, Omm, w0=w0)
    
    def get_corr_factor(self, zl_rep, Omm, w0):
        """
        Args:
          zl_rep (float) : the representative redshift of lens samples
          Omm    (float) : Omega_m at the required cosmology
          w0    (float) : w_de at the required cosmology
        Returns:
          wp_corr (float): the multiplicative correction factor for pimax of wp
        """
        cosmo   = self._get_cosmo(Omm, w0)
        cosmo_meas = self._get_cosmo(self.config['Omm'], self.config['w0'])
        pimax_corr = cosmo.inv_efunc(zl_rep)/cosmo_meas.inv_efunc(zl_rep) # pi = c*Delta z/ H_0*E(z), pimax is proportional to **inverse** of E(z).
        # print(pimax_corr)
        return pimax_corr
    
class rp_meascorr_class:
    def __init__(self, config):
        """
        This class deals with the correction on the projected randial distance.
        
        Args:
          config (dict) : Following keys must be included
            - Omm : Omega_m used for measurement
            - w0  : Equation of state of dark energy used for measurement
        
        https://arxiv.org/abs/2111.10966
        """
        self.config = config
        
    def _get_cosmo(self, Omm, w0):
        return astropy.cosmology.FlatwCDM(100, Omm, w0=w0)
    
    def get_corr_factor(self, zl_rep, Omm, w0):
        cosmo      = self._get_cosmo(Omm, w0)
        cosmo_meas = self._get_cosmo(self.config['Omm'], self.config['w0'])
        r_corr  = cosmo.comoving_distance(zl_rep).value/cosmo_meas.comoving_distance(zl_rep).value        
        return r_corr
