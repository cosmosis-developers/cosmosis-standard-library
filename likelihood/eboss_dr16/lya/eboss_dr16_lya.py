import os
from numpy import log, pi, interp, where, loadtxt, dot, append, linalg, inf, zeros
from cosmosis.datablock import names as section_names
from cosmosis.datablock import option_section
from scipy.interpolate import RegularGridInterpolator

cosmo = section_names.cosmological_parameters
likes = section_names.likelihoods
dist = section_names.distances

c_km_per_s =  299792.458
default_zeff = 2.334
dirname = os.path.dirname(os.path.realpath(__file__))


def setup(options):
    
    section = option_section
    feedback = options.get_bool(section, "feedback", default=False)
    
    print("Lya data")
    print("BAO only: Dm(z)/rd, Dh(z)/rd")
    
    # read alphas and chi2 table
    ly_auto_filename = os.path.join(dirname, 'sdss_DR16_LYAUTO_BAO_DMDHgrid.txt')
    ly_cross_filename = os.path.join(dirname, 'sdss_DR16_LYxQSO_BAO_DMDHgrid.txt')

    DM_auto, DH_auto, like_ratio_auto  = loadtxt(ly_auto_filename).T
    DM_cross, DH_cross, like_ratio_cross  = loadtxt(ly_cross_filename).T

    '''

    # *** Error in the interpolation, maybe because these likes have huge ranges of the order of magnitudes and several data points ***

    like_ratio_auto_interp = interp2d(DM_auto, DH_auto, like_ratio_auto, kind='cubic', fill_value=inf)
    like_ratio_cross_interp = interp2d(DM_cross, DH_cross, like_ratio_cross, kind='cubic', fill_value=inf)

    return like_ratio_auto_interp, like_ratio_cross_interp, feedback

    '''

    # from Arnaud de Mattia & Anderson Souza
    # solution: use RegularGridInterpolator on loglikes

    DM_npoints = 50
    DH_npoints = 50

    # Lya AUTO
    DM_grid = zeros(DM_npoints)
    DH_grid = zeros(DH_npoints)
    prob_grid = zeros((DM_npoints, DH_npoints))
    tmp = loadtxt(ly_auto_filename)
    num = 0
    
    for i in range(DM_npoints):
        for j in range(DH_npoints):
            DM_grid[i], DH_grid[j], prob_grid[i, j] = tmp[num]
            num += 1

    loglike_ratio_auto = log(prob_grid)
    loglike_ratio_auto_interp = RegularGridInterpolator((DM_grid, DH_grid), loglike_ratio_auto, method='linear', bounds_error=False, fill_value=-inf)

    # Lya CROSS
    DM_grid = zeros(DM_npoints)
    DH_grid = zeros(DH_npoints)
    prob_grid = zeros((DM_npoints, DH_npoints))
    tmp = loadtxt(ly_cross_filename)
    num = 0

    for i in range(DM_npoints):
        for j in range(DH_npoints):
            DM_grid[i], DH_grid[j], prob_grid[i, j] = tmp[num]
            num += 1

    loglike_ratio_cross = log(prob_grid)
    loglike_ratio_cross_interp = RegularGridInterpolator((DM_grid, DH_grid), loglike_ratio_cross, method='linear', bounds_error=False, fill_value=-inf)


    return loglike_ratio_auto_interp, loglike_ratio_cross_interp, feedback
    

def execute(block, config):
    
    loglike_ratio_auto_interp, loglike_ratio_cross_interp, feedback = config
        
    # theory
    z = block[dist, 'z']
    rd = block[dist, "rs_zdrag"]
    d_m = block[dist, 'd_m']
    h = c_km_per_s*block[dist, 'h']
    d_h = c_km_per_s/h

    # Get the predicted values of D_M and D_H at the fiducial redshift
    dm_z = interp(default_zeff, z, d_m)
    dh_z = interp(default_zeff, z, d_h)
    dm_z_rd_predicted = dm_z / rd
    dh_z_rd_predicted = dh_z / rd

    loglike_auto = loglike_ratio_auto_interp((dm_z_rd_predicted, dh_z_rd_predicted))
    loglike_cross = loglike_ratio_cross_interp((dm_z_rd_predicted, dh_z_rd_predicted))
    loglike_tot = loglike_auto + loglike_cross

    block[likes,'eboss16_lya_like'] = loglike_tot
    
    if feedback:
        print()
        print('             zeff   pred')
        print('Dm_over_rd: %.3f  %.3f' % (default_zeff, dm_z_rd_predicted))
        print('Dh_over_rd: %.3f   %.3f' % (default_zeff, dh_z_rd_predicted))
        print('loglike:           %.3f' % (loglike_tot))
        print()
    
    return 0

def cleanup(config):
    return 0

