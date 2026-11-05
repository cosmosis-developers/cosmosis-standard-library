r''' 
    python version of FFTLOG by Andrew Hamilton. I am it calling LOG Hankel Transfrom
    This version of the fast Hankle transform is due to 
    Andrew Hamilton (see http://casa.colorado.edu/~ajsh/FFTLog/). 
    The orgrinal algorithm is due to Talman (1978). 
    
    Joseph E. McEwen 
    McEwen Laboratories (c) 2016 
    email: jmcewen314@gmail.com
    
    Please let Joseph E. McEwen aware of any bugs or errors in this code. 
    
    This code is available for anyone to use, but please give approriate reference to 
    Joseph E. McEwen and the authors of the algorithm. 
    
    The Hankel transform in this code is defined as : 
    F(k)= \int_0^\infty f(r) (kr)^q J_\mu(kr) k dr 
    f(r)= \int_0^\infty F(k) (kr)^{-q} J_\mu(kr) r dk . 
    
    Beaware of different definitions, for instance Wikipedie uses the 
    following definitions: 
    F(k)=\int_0^\infty f(r)  J_\mu(kr) r dr
    f(r)= \int_0^\infty F(k)  J_\mu(kr) k dk . 
        
'''
from __future__ import division 

import numpy as np
from numpy.fft import fft, ifft , fftshift, ifftshift , rfft, irfft 
from numpy import exp, log, log10, cos, sin, pi
from scipy.special import gamma 
from time import time 
from numpy import gradient as grad
import sys

log2=log(2)
cut=200  # cutoff to switch to Gamma function limiting case (needed when argument to 
         # gamma function is large) 
import numba

_LANCZOS = np.array([
    0.99999999999980993, 676.5203681218851, -1259.1392167224028,
    771.32342877765313, -176.61502916214059, 12.507343278686905,
    -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7])

@numba.njit(cache=True)
def _lgamma(z):
    # complex log-gamma, Lanczos approximation (g=7, n=9).
    # Only differences of these are used, so branch choices of the log
    # are irrelevant after exponentiation.
    if z.real < 0.5:
        # reflection: Gamma(z) Gamma(1-z) = pi / sin(pi z)
        return np.log(np.pi / np.sin(np.pi * z)) - _lgamma(1.0 - z)
    z = z - 1.0
    x = _LANCZOS[0] + 0j
    for i in range(1, 9):
        x += _LANCZOS[i] / (z + i)
    t = z + 7.5
    return 0.5 * np.log(2 * np.pi) + (z + 0.5) * np.log(t) - t + np.log(x)

@numba.njit(cache=True)
def _g_m_vals(mu, q, cut):
    n = q.size
    g_m = np.zeros(n, dtype=np.complex128)
    for i in range(n):
        qi = q[i]
        if qi == mu + 1 + 0.0j:
            continue  # g_m stays zero
        ap = (mu + 1 + qi) / 2.
        am = (mu + 1 - qi) / 2.
        if abs(qi.imag) + mu + 1 > cut:
            # Stirling series, to higher order
            g_m[i] = np.exp((ap - 0.5) * np.log(ap) - (am - 0.5) * np.log(am) - qi
                            + 1./12 * (1./ap - 1./am)
                            + 1./360. * (1./am**3 - 1./ap**3)
                            + 1./1260 * (1./ap**5 - 1./am**5))
        else:
            g_m[i] = np.exp(_lgamma(ap) - _lgamma(am))
    return g_m

def g_m_vals(mu, q):
    return _g_m_vals(mu, np.asarray(q, dtype=np.complex128), cut)

def log_gamma(z):
    
    z=gamma(z)
    w=log(z)
    x=np.real(w)
    y=np.imag(w)
    return x,y
    
def get_k0(N,mu,q,r0,L,k0):
        
    kr=float(k0*r0)
    delta_L=L/float(N)
    
    x=q + 1j*pi/delta_L
    
    x_plus=(mu+1+x)/2.
    x_minus=(mu+1-x)/2.
        
    rp,phip=log_gamma(x_plus)
    rm,phim=log_gamma(x_minus)
    
    arg=log(2/kr)/delta_L + (phip - phim)/pi 
    iarg=np.rint(arg)
    if ( arg != iarg):
        kr=kr*exp((arg-iarg)*delta_L)
        #kr=kr*exp((arg+iarg)*delta_L)      # Hamilton sign 
    
    return kr 
    
def u_m_vals(m,mu,q,kr,L):

    x=q + 1j*2*pi*m/L
    
    alpha_plus=(mu+1+x)/2.
    alpha_minus=(mu+1-x)/2.
        
    rp, phip=log_gamma(alpha_plus) 
    rm, phim=log_gamma(alpha_minus) 
    
    log_r=q*log2 + rp - rm 
    phi=2*pi*m/L*log(2./kr) + phip - phim 
    
    real_part=exp(log_r)*cos(phi)
    imag_part=exp(log_r)*sin(phi) 
    
    u_m=real_part + 1j*imag_part 
    
    # adjust endpoint, the N/2=m.size point 
    u_m[m.size-1]=np.real(u_m[m.size-1])
    return u_m
    
def u_m_vals_new(m,mu,q,kr,L):

    omega=1j*2*pi*m/L

    x=q + omega
       
    two_part=2**x 
       
    U_mu=2**x*g_m_vals(mu,x)
    
    u_m=(kr)**(-omega)*U_mu
    
    u_m[m.size-1]=np.real(u_m[m.size-1])
    
    return u_m 
    
def fft_log(k,f_k,q,mu,kr=None):


    if ((q+mu) < -1) :
        print('Error in reality condition for Bessel function integration.')
        print(' q+mu is less than -1.')
        print('See Abramowitz and Stegun. Handbook of Mathematical Functions pg. 486')
        
    
    if ( q > 1/2.) :
        print('Error in reality condition for Bessel function integration.')
        print(' q is greater than 1/2')
        print('See Abramowitz and Stegun. Handbook of Mathematical Functions pg. 486')

        
                
    N=f_k.size
    delta_L=(log(np.max(k))-log(np.min(k)))/float(N-1)
    #delta_L10=(np.log10(np.max(k))-np.log10(np.min(k)))/(N-1)
    L=(log(np.max(k))-log(np.min(k)))
        
    # find a better way to check if it is evenly spaced in log 
    diff=np.diff(np.log(k))
    diff=np.diff(diff)
    if (np.sum(diff) >=1e-10):
        raise ValueError('You need to send in data that is sampled evenly in logspace')
        
    
    log_k0=log(k[N//2])
    k0=exp(log_k0)
    
    # Fourier transform input data 
    # get m values, shifted so the zero point is at the center
    
    c_m=rfft(f_k)
    m=np.fft.rfftfreq(N,d=1.)*float(N)
    # make r vector 
    #kr=get_k0(float(N),mu,q,1/k0,L,k0)
    if kr is None:
        kr=mu+0.5
    r0=kr/k0
    log_r0=log(r0)
    
    m=np.fft.rfftfreq(N,d=1.)*float(N)
    m_r=np.arange(-N//2,N//2)
    m_shift=np.fft.fftshift(m_r)
    
    
    #s-array 
    s=delta_L*(-m_r)+log_r0     
    id=m_shift
    r=10**(s[id]/log(10))
    
    #m_shift=np.fft.fftshift(m)
    
    # get h array   
    h=delta_L*m + log_k0
        
    u_m=u_m_vals_new(m,mu,q,kr,L)
    #u_m=u_m_vals(m,mu,q,kr,L) old version will crash for large data set 

    b=c_m*u_m
        
    A_m=irfft(b)
    
    A=A_m[id]
    
    # reverse the order 
    A=A[::-1]
    r=r[::-1]
    
    if (q!=0):
        A=A*(r)**(-float(q))
        
    return r, A 

##########################################################################################
# End of fftlog algorithm 



##########################################################################################
# function specific for power spectrum to correlation function (and vice versus) in 
# cosmology 
def k_to_r(k,f_k,alpha_k=1.5, beta_r=-1.5, mu=.5, pf=(2*pi)**(-1.5),q=0):
    
    # module to calculate Hankel Transform
    # \int_0^\infty dk r A(k) J_mu(kr), via fftlog algorithm
    # Common application is for power spectrum:
    # \xi(r)= \int dk k^2 /(2 \pi^2) \sin(kr)/kr P(k) 
    # in which case 
    # alpha_k=1.5
    # beta_r=-1.5
    # mu=.5 
    # pf=(2*np.pi)**(-1.5)
    
    f_k=k**alpha_k*f_k
    
    r, A=fft_log(k,f_k,q,mu)

    f_r=pf*A*r**beta_r 
    
    return r, f_r 
    
def r_to_k(r,f_r,alpha_k=-1.5, beta_r=1.5, mu=.5, pf=4*pi*np.sqrt(pi/2.),q=0):
    
    # module to calculate Hankel Transform
    # \int_0^\infty dr k A(r) J_mu(kr), via fftlog algorithm
    # Common application is for correlation function:
    # P(k)= 2 pi \int dr r^2  \sin(kr)/kr xi(r) 
    # in which case 
    # alpha_k=-1.5
    # beta_r=1.5
    # mu=.5 
    # pf=4 pi *sqrt(pi/2)
    
    f_r=r**beta_r*f_r
    k, A=fft_log(r,f_r,q,mu)
    
    f_k=pf*A*k**alpha_k 
    return k, f_k
    
