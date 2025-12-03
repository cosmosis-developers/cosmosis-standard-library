'''
The module interfaces with the FAST-PT code, which has a separately maintained repository.
For ease of use and to ensure consistency FAST-PT will be included in the
cosmosis-standard-library. Currently, FAST-PT v2.0 is included.

FAST-PT performs various 1-loop PT calculations,
including biasing, IA, and RSD.
See McEwen et al 2016 (arXiv:1603.04826) and Fang et al 2016 (arXiv:1609.05978).
Contact Jonathan Blazek with questions.
'''

# TO DO
# Streamline code
# Remove extrapolate in interp1d due to scipy compatibility
#
import scipy.integrate
scipy.integrate.trapz = scipy.integrate.trapezoid
from cosmosis.datablock import names, option_section
from fastpt import FASTPT
try:
    from fastpt.P_extend import k_extend
except ModuleNotFoundError:
    from fastpt.utils.P_extend import k_extend
import numpy as np
from time import time

# hack because fastpt not yet updated for numpy 1.24
if not hasattr(np, 'int'):
    np.int = int

from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline as intspline
#import matplotlib as plt

def setup(options):
    do_dd_spt = options.get_bool(option_section, 'do_dd_spt', False)
    do_ia = options.get_bool(option_section, 'do_ia', False)
    do_bias = options.get_bool(option_section, 'do_bias', False)
    do_rsd = options.get_bool(option_section, 'do_rsd', False)
    do_ia_tt = options.get_bool(option_section, 'do_ia_tt', False)
    do_ia_ta = options.get_bool(option_section, 'do_ia_ta', False)
    do_ia_mix = options.get_bool(option_section, 'do_ia_mix', False)
    bias_section = options.get_string(option_section, 'bias_section', 'bias')
    output_nl_grid = options.get_bool(option_section, "output_nl_grid", True)
    only_terms = options.get_bool(option_section, "only_terms", False) #only save individual PT terms to block, not e.g. galaxy and galaxy-matter P(k)s
    always_init = options.get_bool(option_section, "always_init", False)
    k_res_fac = options.get_double(option_section, "k_res_fac", 4.0)
    low_extrap = options.get_double(option_section, "low_extrap", -5)
    high_extrap = options.get_double(option_section, "high_extrap", 3)
    C_window = options.get_double(option_section, "C_window", 0.75)
    n_pad_fac = options.get_double(option_section, "n_pad_fac", 1.0)
    verbose = options.get_bool(option_section, "verbose", False)
    if options.has_value(option_section, "P_window"):
        P_window = options.get_double(option_section, "P_window")
    else:
        P_window = None

    #####################    
    do_fastpt=False
    fpt_config = {
        'do_dd_spt':do_dd_spt,
        'do_ia':do_ia,
        'do_bias':do_bias, 
        'do_rsd':do_rsd,
        'do_ia_tt':do_ia_tt,
        'do_ia_ta':do_ia_ta,
        'do_ia_mix':do_ia_mix
    }

    n_true = sum(fpt_config.values())
    if n_true > 0:
        do_fastpt=True
        print('You have asked for quantities from FAST-PT. FAST-PT will be run.')
    else:
        print('WARNING: You have NOT asked for quantities from FAST-PT. FAST-PT will NOT be run.')
        
    config = {
        'do_fastpt':do_fastpt,
        'bias_section':bias_section,
        'output_nl_grid':output_nl_grid,
        'only_terms':only_terms,
        'always_init':always_init,
        'k_res_fac':k_res_fac,
        'low_extrap':low_extrap,
        'high_extrap':high_extrap,
        'P_window':P_window,
        'C_window':C_window,
        'n_pad_fac':n_pad_fac,
        'verbose':verbose
    }
    config.update(fpt_config)

    return config

def execute(block, config):
    verbose = config['verbose']
    if config['do_fastpt'] == False:
        if verbose: print('WARNING: You have NOT asked for quantities from FAST-PT. FAST-PT will NOT be run.')
        return 0 # skip the rest of execute
    t3=time()
    need_reinit = False
    lin = names.matter_power_lin
    nl = names.matter_power_nl

    z, k0, Pkz = block.get_grid(lin, "z", "k_h", "P_k")
    znl, knl1, Pkznl = block.get_grid(nl, "z", "k_h", "P_k")


    # note that extension, if needed, must be done on Plin at every step.
    if config['always_init']:
        need_reinit = True
        if verbose: print("You have chosen to initialize FAST-PT at each step.")
    elif 'fastpt_kinit' in config.keys():
        try:
            np.testing.assert_allclose(config['k0'], k0, rtol=1e-04,
            err_msg='klin values do not match.', verbose=True)
            np.testing.assert_allclose(config['knl1'], knl1, rtol=1e-04,
            err_msg='knl values do not match.', verbose=True)
        except:
            need_reinit = True
            if verbose: print("WARNING: k values didn't match, so FAST-PT will re-initialize.")
        else:
            if verbose: print("FAST-PT is already initialized at the correct k values.")
    else:
        need_reinit = True
        if verbose: print("FAST-PT has not yet been initialized - this will be done now.")
    
    k0log=np.log(k0)
    knl1log=np.log(knl1)
    P0 = Pkz[0,:]
    ind=np.where(k0>0.03)[0][0]
    #note minor scale-dependence in growth factor.
    ##use z values from linear CAMB
    Growth2 = Pkz[:,ind]/Pkz[0,ind]

    #Growth2 = Pkznl[:,ind]/Pkz[0,ind]
    #test that nl and lin have same z values
    msg='lin and nl power spectra are calculated at different z values.'
    np.testing.assert_array_almost_equal(z, znl, decimal=4, err_msg=msg, verbose=False)


#   Put P_lin on a log k grid, required for FAST-PT steps
    #ringing in results appears to be sensitive to sampling and/or interpolation
    nk=int(config['k_res_fac']*len(k0)) # higher res increases runtime, decreases potential ringing at low-k
    if (nk % 2) != 0:
        nk +=1 # make sure nk is even
    #kmin=np.log10(k0[0]); kmax=np.log10(k0[-1])    
    eps=0.    ## This interpolation should be at the input bounds. Extrapolate used to avoid failure due to numerical noise. No actual extrapolation is done. We could handle this using a margin set by eps, or by interpolation which allows for extrapolation.
    kmin=np.log10((1.+eps)*k0[0]); kmax=np.log10((1.-eps)*k0[-1])
    k1=np.logspace(kmin,kmax,nk)
    k1log=np.log(k1)
#    pinterp=interp1d(k0log,np.log(P0),bounds_error=False, fill_value="extrapolate") # this is the old syntax. Doesn't work with older versions of scipy. CONFIRM consistent results with intspline.
    pinterp=intspline(k0log,np.log(P0))

    P1=np.exp(pinterp(k1log))


# Extend the P_lin range to outside the nl range. uses power law extensions from FAST-PT
##This step is not needed if we decide to output at the lin k grid rather than the NL k grid.
    eps2=1e-7 #margin to account for numerical noise
    if (knl1[0]*(1.+eps2)<k1[0]) or (knl1[-1]*(1.-eps2)>k1[-1]):
        if verbose:
            print('extending from old (knl1[0],k1[0],knl1[-1],k1[-1])')
            print(knl1[0],k1[0],knl1[-1],k1[-1])
            print('to:')
        EK1=k_extend(k1,np.log10(knl1[0]),np.log10(knl1[-1]))
#        EK1=k_extend(k1,-6.,3.)#testing option
        k1=EK1.extrap_k()
        if verbose: print(knl1[0],k1[0],knl1[-1],k1[-1])
        P1=EK1.extrap_P_low(P1)
        P1=EK1.extrap_P_high(P1)
#        kmin=np.log10(knl1[0]); kmax=np.log10(knl1[-1])
        kmin=np.log10(k1[0]); kmax=np.log10(k1[-1])

        if verbose:
            print('Extended krange:',kmin,kmax)
            print('NL k range:',np.log10(knl1[0]),np.log10(knl1[-1]))


# We currently interpolate each output onto the nl grid later.
# This might be more accurate due to the potentially larger k range in the lin k grid,
# but it increases runtime due to more interpolation steps.
# Alternatively, we could interpolate Plin onto the nl grid here.

# is the NL grid always log spaced? if so, we could everything on the NL grid.
# or, we could do a k extend and interpolation for every z value for P_NL
#    pinterp=interp1d(np.log(k1),np.log(P1))
#    #note: machine precision can cause issues with interpolation at the boundaries
#    k=np.logspace(kmin,kmax,nk)
#    P=np.exp(pinterp(klog))
    
    k=k1
    klog=np.log(k) #this is used for log interpolation later, so uses natural log.
    P=P1
    
    t4=time()
    if verbose: print('Preparing arrays for FAST-PT took',t4-t3)

    # interpolation in lin-lin space
    #NOTE: tends to cause ringing at high-k
#    pinterp=interp1d(k1,P1)
#    kmin=np.log10(1.00001*k1[0]); kmax=np.log10(0.999999*k1[-1])
#    nk=5*len(k1)
#    k=np.logspace(kmin,kmax,nk)
#    P=pinterp(k)
    

    #Growth2 = Pkz[:,0]/Pkz[0,0]
    #Note that the full Boltzmann solution from CAMB includes a small
    #scale dependence in the growth factor, at the ~0.5% level.
    #It is present on the largest scales, so we choose a larger
    #value of k to determine the growth factor.

    if need_reinit:
        t5=time()
        #specify which FAST-PT calculations to initialize
        to_do=[]
        if config['do_bias']:
            to_do.append('dd_bias') 
        if config['do_dd_spt']:
            to_do.append('one_loop_dd')
        if config['do_ia']:
            to_do.append('IA_all')
        if config['do_ia_tt']:
            to_do.append('IA_tt')
        if config['do_ia_ta']:
            to_do.append('IA_ta')
        if config['do_ia_mix']:
            to_do.append('IA_mix')
        if config['do_rsd']:
    	    to_do.append('RSD')
        if verbose: print('intializing FAST-PT on a new k grid with the following to_do list:',to_do)
        # bias parameter and padding length 
        #nu=-2 #shouldn't be required in v2
        n_pad=int(config['n_pad_fac']*len(k))
        config['fastpt_kinit'] = FASTPT(k,to_do=to_do,low_extrap=config['low_extrap'],high_extrap=config['high_extrap'],n_pad=n_pad) 
        config['k0'] = k0
        config['knl1'] = knl1
        t6=time()
        if verbose: print('FAST-PT kgrid initialization took',t6-t5)

    #P_window=np.array([0.2,0.2])
    P_window = config['P_window']
    C_window = config['C_window']
    fastpt = config['fastpt_kinit']
    Plinfpt = np.outer(Growth2,P)
    #pre-calculate this now, and then you don't need to run do_bias.
    #This version won't be filtered (it matches the input Plin).

    if config['do_bias']:
        t7=time()
        bias_fpt=fastpt.one_loop_dd_bias(P,P_window=P_window,C_window=C_window)
        Pkz2 = np.outer(Growth2,P) #test the new growth factor
        Plinfpt = np.outer(Growth2,bias_fpt[1])
        one_loopkz = np.outer(Growth2**2,bias_fpt[0])
        Pd1d2 = np.outer(Growth2**2,bias_fpt[2])
        Pd2d2 = np.outer(Growth2**2,bias_fpt[3])
        Pd1s2 = np.outer(Growth2**2,bias_fpt[4])
        Pd2s2 = np.outer(Growth2**2,bias_fpt[5])
        Ps2s2 = np.outer(Growth2**2,bias_fpt[6])
        sig4kz = np.outer(Growth2**2,bias_fpt[7]*np.ones_like(bias_fpt[0]))
        P1Lkz= Pkz2 + one_loopkz
        
        ####
        output_nl_grid=config['output_nl_grid']
        if output_nl_grid:

		    # interpolate to nl k grid. Interpolation is different for terms that can have negative values.
            if verbose:
                print('outputing all fast-pt results on NL k grid as well.')
                print('klog bounds:',klog[0],klog[-1])
                print('knl1log bounds:',knl1log[0],knl1log[-1])

            temp=intspline(klog,bias_fpt[2])
            Pd1d2_o = np.outer(Growth2**2,temp(knl1log))

            temp=intspline(klog,np.log(bias_fpt[3]))
            Pd2d2_o = np.outer(Growth2**2,np.exp(temp(knl1log)))

            temp=intspline(klog,bias_fpt[4])
            Pd1s2_o = np.outer(Growth2**2,temp(knl1log))

            temp=intspline(klog,bias_fpt[5])
            Pd2s2_o = np.outer(Growth2**2,temp(knl1log))

            temp=intspline(klog,np.log(bias_fpt[6]))
            Ps2s2_o = np.outer(Growth2**2,np.exp(temp(knl1log)))
            sig4kz_o = np.outer(Growth2**2,bias_fpt[7]*np.ones_like(knl1))
        
        ####
        #Power law in z bias model (constant in z if b<i>_alpha not set in values file)
        bias_section=config['bias_section']
        b1_z0 = block[bias_section, "b_1"]
        b2_z0 = block[bias_section, "b_2"]
        bs_z0 = block[bias_section, "b_s"]

        b1_alpha = block.get_double(bias_section, "b1_alpha", 0.)
        b2_alpha = block.get_double(bias_section, "b2_alpha", 0.)
        bs_alpha = block.get_double(bias_section, "bs_alpha", 0.)

        z0_1 = block.get_double(bias_section, "b1_z0", 0.)
        z0_2 = block.get_double(bias_section, "b2_z0", 0.)
        z0_s = block.get_double(bias_section, "bs_z0", 0.)
        
        K,Z = np.meshgrid(k,z) #why have a k-dimension for the bias?
        b1 = b1_z0*((1+Z)/(1+z0_1))**b1_alpha#np.ones_like(Z) seems unnecessary
        b2 = b2_z0*((1+Z)/(1+z0_2))**b2_alpha
        bs = bs_z0*((1+Z)/(1+z0_s))**bs_alpha
        

        ###
        # BIAS CONVENTION. See, e.g. https://arxiv.org/pdf/1405.1447v4.pdf
        # delta_g = b1*delta_m + (1/2)*b2*(delta_m^2-<delta_m^2>) + (1/2)*b_s*(s^2-<s^2>)
        # fiducial quadratic bias values:
        # b2 = 0.9*(b1-1)^2-0.5 (this is a numerical fit to simulation data, but a relationship of this form is motivated in the spherical collapse picture
        # bs = (-4/7)(b1-1)
        #
        ###

        #NOTE: With quadratic bias models, the 'shot noise' becomes a free parameter and can be non-zero for cross-correlations between two different samples.

        # galaxy power spectrum (for bin AUTO correlations)
        Pggsub = (b1**2*Pkz2 + b1*b2*Pd1d2 + (1./4)*b2*b2*(Pd2d2-2.*sig4kz) + b1*bs*Pd1s2 +
            (1./2)*b2*bs*(Pd2s2-4./3*sig4kz) + (1./4)*bs*bs*(Ps2s2-8./9*sig4kz))
        
        Pgg = (b1**2*Pkz2 + b1*b2*Pd1d2 + (1./4)*b2*b2*(Pd2d2) + b1*bs*Pd1s2 +
            (1./2)*b2*bs*(Pd2s2) + (1./4)*bs*bs*(Ps2s2))

        # galaxy-matter cross power spectrum
        Pmg = b1*Pkz2 + (1./2)*b2*Pd1d2 + (1./2)*bs*Pd1s2
        
        # note that the fastpt block uses Plin for the linear
        #  bias terms in P_mg and P_gg
        # uses a log k grid with the lin k bounds.
        block.put_grid('fastpt',"z",z,"k_h",k,'Plin',Plinfpt)
        block.put_grid('fastpt',"z",z,"k_h",k,'Pd1d2',Pd1d2)
        block.put_grid('fastpt',"z",z,"k_h",k,'Pd2d2',Pd2d2)
        block.put_grid('fastpt',"z",z,"k_h",k,'Pd1s2',Pd1s2)
        block.put_grid('fastpt',"z",z,"k_h",k,'Pd2s2',Pd2s2)
        block.put_grid('fastpt',"z",z,"k_h",k,'Ps2s2',Ps2s2)
        block.put_grid('fastpt',"z",z,"k_h",k,'Pgg',Pgg)
        block.put_grid('fastpt',"z",z,"k_h",k,'Pggsub',Pggsub)
        block.put_grid('fastpt',"z",z,"k_h",k,'Pmg',Pmg)
        block.put_grid('fastpt',"z",z,"k_h",k,'sig4kz',sig4kz)
        block.put_grid('fastpt',"z",z,"k_h",k,'one_loopkz',one_loopkz)
        block.put_grid('fastpt',"z",z,"k_h",k,'P1Lkz',P1Lkz)
        block.put_grid('fastpt',"z",z,"k_h",k,'Pkz2',Pkz2)
        
        # put galaxy-galaxy and galaxy-matter power spectra into the appropriate bits of the data block
        # First, must interpolate back to original k values (choose lin or nl)
        if not config['only_terms']:
            block.put_grid("galaxy_power", "z", z, "k_h", k, "p_k", Pgg)
            block.put_grid("galaxy_power_sublowk", "z", z, "k_h", k, "p_k", Pggsub)
            block.put_grid("matter_galaxy_power", "z", z, "k_h", k, "p_k", Pmg)

        if output_nl_grid:

            K,Z = np.meshgrid(knl1,z)
            b1 = b1_z0*((1+Z)/(1+z0_1))**b1_alpha
            b2 = b2_z0*((1+Z)/(1+z0_2))**b2_alpha
            bs = bs_z0*((1+Z)/(1+z0_s))**bs_alpha

            Pgg_o = (b1**2*Pkznl + b1*b2*Pd1d2_o + (1./4)*b2*b2*(Pd2d2_o) + b1*bs*Pd1s2_o +
                (1./2)*b2*bs*(Pd2s2_o) + (1./4)*bs*bs*(Ps2s2_o))
            Pggsub_o = (b1**2*Pkznl + b1*b2*Pd1d2_o + (1./4)*b2*b2*(Pd2d2_o-2.*sig4kz_o) + b1*bs*Pd1s2_o +
                (1./2)*b2*bs*(Pd2s2_o-4./3*sig4kz_o) + (1./4)*bs*bs*(Ps2s2_o-8./9*sig4kz_o))
            Pmg_o = b1*Pkznl + (1./2)*b2*Pd1d2_o + (1./2)*bs*Pd1s2_o
            if not config['only_terms']:
                block.put_grid("galaxy_power_o", "z", z, "k_h_nl", knl1, "p_k", Pgg_o)
                block.put_grid("galaxy_power_sublowk_o", "z", z, "k_h_nl", knl1, "p_k", Pggsub_o)
                block.put_grid("matter_galaxy_power_o", "z", z, "k_h_nl", knl1, "p_k", Pmg_o)
            block.put_grid('fastpt',"z",z,"k_h_nl",knl1,'Pd1d2_o',Pd1d2_o)
            block.put_grid('fastpt',"z",z,"k_h_nl",knl1,'Pd2d2_o',Pd2d2_o)
            block.put_grid('fastpt',"z",z,"k_h_nl",knl1,'Pd1s2_o',Pd1s2_o)
            block.put_grid('fastpt',"z",z,"k_h_nl",knl1,'Pd2s2_o',Pd2s2_o)
            block.put_grid('fastpt',"z",z,"k_h_nl",knl1,'Ps2s2_o',Ps2s2_o)
            block.put_grid('fastpt',"z",z,"k_h_nl",knl1,'Pgg_o',Pgg_o)
            block.put_grid('fastpt',"z",z,"k_h_nl",knl1,'Pggsub_o',Pggsub_o)
            block.put_grid('fastpt',"z",z,"k_h_nl",knl1,'Pmg_o',Pmg_o)
            block.put_grid('fastpt',"z",z,"k_h_nl",knl1,'sig4kz_o',sig4kz_o)

        t8=time()
        if verbose: print('FAST-PT nonlinear bias steps took',t8-t7)    


    if config['do_ia']:
        t9=time()
        IA_tt=fastpt.IA_tt(P,P_window=P_window,C_window=C_window)
        tt_EE=np.outer(Growth2**2,IA_tt[0])
        tt_BB=np.outer(Growth2**2,IA_tt[1])
        IA_ta=fastpt.IA_ta(P,P_window=P_window,C_window=C_window)
        ta_dE1=np.outer(Growth2**2,IA_ta[0])#confirm growth dependence, etc
        ta_dE2=np.outer(Growth2**2,IA_ta[1])
        ta_EE=np.outer(Growth2**2,IA_ta[2])
        ta_BB=np.outer(Growth2**2,IA_ta[3])
        IA_mix=fastpt.IA_mix(P,P_window=P_window,C_window=C_window)
        mix_A=np.outer(Growth2**2,IA_mix[0])
        mix_B=np.outer(Growth2**2,IA_mix[1])
        mix_D_EE=np.outer(Growth2**2,IA_mix[2])
        mix_D_BB=np.outer(Growth2**2,IA_mix[3])

        if not config['do_bias']:
            margin=1.0-1e-6
            sig4=IA_tt[0][0]*margin
            #this is required to subtract off the k->0 limit. FAST-PT calculates the relevant value in the bias
            #module, but not in the IA module. If only IA is being used, we can just use the minimum k value
            #returned, with a small margin to prevent a zero value.
            # Do we want to use the min k value, or is that affected by ringing?
            # This is not actually the sig4 value, but rather some factor multiplied by it. UPDATE

        #we may want to change this to nonlinear k binning at some point, although perhaps could be done downstream
        block.put_grid('fastpt',"z",z,"k_h",k,'Plin',Plinfpt)
        block.put_grid('fastpt',"z",z,"k_h",k,'P_tt_EE',tt_EE)
        block.put_grid('fastpt',"z",z,"k_h",k,'P_tt_BB',tt_BB)
        block.put_grid('fastpt',"z",z,"k_h",k,'P_ta_dE1',ta_dE1)
        block.put_grid('fastpt',"z",z,"k_h",k,'P_ta_dE2',ta_dE2)
        block.put_grid('fastpt',"z",z,"k_h",k,'P_ta_EE',ta_EE)
        block.put_grid('fastpt',"z",z,"k_h",k,'P_ta_BB',ta_BB)
        block.put_grid('fastpt',"z",z,"k_h",k,'P_mix_A',mix_A)
        block.put_grid('fastpt',"z",z,"k_h",k,'P_mix_B',mix_B)
        block.put_grid('fastpt',"z",z,"k_h",k,'P_mix_D_EE',mix_D_EE)
        block.put_grid('fastpt',"z",z,"k_h",k,'P_mix_D_BB',mix_D_BB)
        t10=time()
        if verbose:
            print('FAST-PT IA steps took',t10-t9)

##########

    if verbose:
        print('FAST-PT done')
    return 0

def cleanup(config):
    pass
