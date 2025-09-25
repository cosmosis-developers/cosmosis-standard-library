"""
Author: Sunao Sugiyama
Last edit: 2023.05.17

Code reviewed by Tianqing Zhang in Oct 2023
"""
import numpy as np
from scipy import integrate
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy.special import eval_legendre
from fftlog import hankel, fftlog

class minimalbias_class:
    """
    Gives the g-g lensing, dSigma, and the projected galaxy correlation function, wp.
    These observables is independent from the source redshift and only the function of lens redshift.
    
    The observational corrections are applied outside this class.
        - cosmological dependence of measurements (AP-like effect), this changes pimax and R.
        - photo-z correction of dSigma via Sigma_crit factor in the estimator
    """
    def __init__(self, config=None):
        # config
        self.config = {'do_Kaiser': True, 
                       'verbose'  : True}
        self.config.update(config or {})
    
    def set_param(self, Omm, b1):
        """
        Set parameters. Parameter depenence of the observable on other 
        cosmological parameters are all imprinted in power spectra and f(z).
        """
        self.param = {'Omm': Omm, 'b1':b1}
        
    def set_pk_lin_data(self, z, k, pklin):
        """
        set linear matter power spectrum
        """
        self.pk_lin_data = [z, k, pklin]
        
    def set_pk_nlin_data(self, z, k, pknonlin):
        """
        set nonlinear matter power spectrum
        """
        self.pk_nlin_data = [z, k, pknonlin]
        
    def set_z2f(self, z, f):
        """
        set mapping function from redshift z to logarithmic graowth rate f(z).
        """
        self.z2f = ius(z, f)
        
    def get_pk_lin_at_z(self, z_in):
        """
        Gives linear matter power spectrum which is linearly interpolated over redshift
        """
        z, k, pk = self.pk_lin_data
        idx0 = np.argwhere(z<z_in)[-1][0]
        idx1 = idx0+1
        z0, pk0 = z[idx0], pk[idx0, :]
        z1, pk1 = z[idx1], pk[idx1, :]
        return k, pk0 + (pk1-pk0) / (z1-z0) * (z_in - z0)
        
    def get_pk_nlin_at_z(self, z_in):
        """
        Gives nonlinear matter power spectrum which is linearly interpolated over redshift
        """
        z, k, pk = self.pk_nlin_data
        # print(np.argwhere(z<z_in))
        idx0 = np.argwhere(z<z_in)[-1][0]
        idx1 = idx0+1
        z0, pk0 = z[idx0], pk[idx0, :]
        z1, pk1 = z[idx1], pk[idx1, :]
        
        
        # print(z_in, z0, z1)
        return k, pk0 + (pk1-pk0) / (z1-z0) * (z_in - z0)
    
    def get_ds(self, z, rp, dlnrp, N_extrap_low=None, N_extrap_high=None):
        """
        rp     (array): array of radial bins in unit of Mpc/h, need to be corrected for the measurement effect.
        """
        k, pk = self.get_pk_nlin_at_z(z)
        # np.savetxt('./output/pk_at_%f_camb.txt'%z, np.array([k,pk]))
        if N_extrap_low is None:
            N_extrap_low = k.size
        if N_extrap_high is None:
            N_extrap_high = k.size
        b1      = self.param['b1']
        rhocrit = 2.77536627e11 # h^2 M_sun Mpc^-3
        rhom0   = rhocrit*self.param['Omm']
        myhankel = hankel(k, k**2*pk, 1.01, 
                          N_extrap_low=N_extrap_low, 
                          N_extrap_high=N_extrap_high)
        if dlnrp > 0:
            D = 2 # dimension
            R, ds = myhankel.hankel_binave(2, dlnrp, D)
        else:
            R, ds = myhankel.hankel(2)
        ds *= b1*rhom0/(2.0*np.pi)
        ds /= 1e12 # [h M_sun/pc^2]
        return ius(R, ds)(rp)
        
    def _get_xigg_n(self, k, pk, n, N_extrap_low=None, N_extrap_high=None):
        if N_extrap_low is None:
            N_extrap_low = k.size
        if N_extrap_high is None:
            N_extrap_high = k.size
        b1      = self.param['b1']
        myfftlog = fftlog(k, k**3*pk, 1.01, 
                          N_extrap_low=N_extrap_low, 
                          N_extrap_high=N_extrap_high)
        r, xi = myfftlog.fftlog(n)
        xi *= b1**2/(2*np.pi**2)
        return r, xi
        
    def _get_wp_pimax(self, k, pk, pimax, dlnrp):
        r, xi = self._get_xigg_n(k, pk, 0, N_extrap_low=0, N_extrap_high=0 )
        
        rpi = np.logspace(-3, np.log10(pimax), 300)
        rp2d, rpi2d = np.meshgrid(r, rpi)
        s = (rp2d**2+rpi2d**2)**0.5
        xi2d = ius(r, xi, ext=3)(s)

        
        wp = 2*integrate.simps(xi2d, rpi, axis=0)
        
        if dlnrp>0.0:
            wp = binave_array(r, wp, dlnrp)
        
        return r, wp
        
    def _get_wp_pimaxinf(self, k, pk, dlnrp, N_extrap_low=None, N_extrap_high=None):
        if N_extrap_low is None:
            N_extrap_low = k.size
        if N_extrap_high is None:
            N_extrap_high = k.size
        b1      = self.param['b1']
        myhankel = hankel(k, pk*k**2, 1.01, 
                          N_extrap_low, 
                          N_extrap_high)
        if dlnrp == 0.0:
            R, wp = myhankel.hankel(0)
        else:
            R, wp = myhankel.hankel_binave(0, dlnrp, 2)
        wp *= b1**2/(2.0*np.pi)
        return R, wp
    
    def _get_wp_Kaiser(self, k, pk, fz, pimax):
        r, xi0 = self._get_xigg_n(k, pk, 0)
        r, xi2 = self._get_xigg_n(k, pk, 2)
        r, xi4 = self._get_xigg_n(k, pk, 4)
        xi2 = -xi2 # fftlog does not include (-i)^n factor

        # calculate beta
        b1= self.param['b1']
        beta = fz/b1

        wp_rsd = _get_wp_aniso(r, xi0, xi2, xi4, beta, r, pimax)
        
        return r, wp_rsd
    
    def get_wp(self, z, rp, dlnrp, pimax):
        """
        rp     (array): array of radial bins in unit of Mpc/h, need to be corrected for the measurement effect.
        pimax (float): projection length in unit of Mpc/h, need to be corrected for the measurement effect.
        """
        if not self.config['do_Kaiser']:
            if pimax == 'inf':
                k, pknonlin = self.get_pk_nlin_at_z(z)
                r, wp = self._get_wp_pimaxinf(k, pknonlin, dlnrp)
            elif isinstance(pimax, (int, float)):
                k, pknonlin = self.get_pk_nlin_at_z(z)
                r, wp = self._get_wp_pimax(k, pknonlin, pimax, dlnrp)
            else:
                raise ValueError('pimax must be int or "inf".')
        else:
            if pimax == 'inf':
                raise ValueError('Cannot apply Kaiser with pimax="inf".')
            elif isinstance(pimax, (int, float)):
                k, pknonlin = self.get_pk_nlin_at_z(z)
                fz = self.z2f(z)
                r, wp = self._get_wp_pimax(k, pknonlin, pimax, 0.0)
                k, pklin = self.get_pk_lin_at_z(z)
                r_aniso, wp_aniso = self._get_wp_Kaiser(k, pklin, fz, pimax)
                r_iso  , wp_iso   = self._get_wp_pimax(k, pklin, pimax, 0.0)
                assert np.all(r_aniso==r) and np.all(r_iso==r), 'r binning do not match.'
                wp *= wp_aniso/wp_iso
            else:
                raise ValueError('pimax must be int or "inf".')
            if dlnrp > 0.0:
                wp = binave_array(r, wp, dlnrp)
        
        return ius(r, wp)(rp)
        
def binave_array(x, y, dlnx, D=2, nbin=100):
    """
    Assumes dlnx << np.diff(np.log(x)).
    Performs the forward bin average in dimension D.
    ::math::
        \\bar{y} = \\frac{1}{d\\ln x} \\int_{\\ln x}^{\\ln x+d\\ln x} x^D y(x)
    """
    X = np.linspace(0.0, dlnx, nbin)
    x2d, X2d = np.meshgrid(x,X)
    arg = x2d*np.exp(X2d)
    y2d = ius(x, y)(arg)
    
    nom = integrate.simps(arg**D, X, axis=0)
    
    ybar = integrate.simps(y2d*arg**D, X, axis=0)/nom
    
    return ybar
        
def _get_wp_aniso(r, xi0, xi2, xi4, beta, rp_in, pimax):
    # Numerator of Eq. (48) of arxiv: 1206.6890 using mutipole expansion of anisotropic xi including Kaiser effect in Eq. (51) of the same paper.
    dlnrp_min = 0.01 # bin size of dlnrp enough to obtain 0.01 %
    if np.log10(rp_in[1]/rp_in[0]) < dlnrp_min:
        print('Input rp is dense. Using more sparse rp to calculate wp_aniso.')
        # calcurate wp_aniso on more sparse rp and then interpolate it to obtain wp_aniso on rp_in.
        rp = 10**np.arange(np.log10(rp_in.min()), np.log10(rp_in.max()), dlnrp_min)
        interpolate = True
    else:
        rp = rp_in
        interpolate = False

    rpi = np.logspace(-3, np.log10(pimax), 300) # Ok binning for 1% accuracy.

    rp2, rpi2 = np.meshgrid(rp, rpi)

    s = (rp2**2+rpi2**2)**0.5
    mu= rpi2/s

    xi0s = (1+2.0/3.0*beta+1.0/5.0*beta**2)*ius(r, xi0)(s)
    xi2s = (4.0/3.0*beta+4.0/7.0*beta**2)*ius(r, xi2)(s)
    xi4s = 8.0/35.0*beta**2*ius(r, xi4)(s)

    p0 = 1
    p2 = eval_legendre(2, mu)
    p4 = eval_legendre(4, mu)

    xi_aniso = 2*(xi0s*p0+xi2s*p2+xi4s*p4)

    wp_aniso = integrate.simps(xi_aniso, rpi, axis=0)

    if interpolate:
        wp_aniso = ius(rp, wp_aniso)(rp_in)
    
    return wp_aniso
    
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
    