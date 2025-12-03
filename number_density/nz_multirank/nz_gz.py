'''
Conversion of n(z) to g(chi) and back again

Credit to Nicolas Tessore.

cosmo should be an astropy.cosmology object

e.g. cosmo = astropy.cosmology.Planck15
'''

from astropy import cosmology

def nz_to_gchi(z, nz, cosmo=cosmology.Planck15):
    from numpy import apply_along_axis, gradient, multiply, newaxis
    try:
        from scipy.integrate import cumulative_trapezoid
    except ImportError:
        from scipy.integrate import cumtrapz as cumulative_trapezoid
    chi = cosmo.comoving_distance(z).value
    dchi = apply_along_axis(gradient, -1, chi)
    dz = apply_along_axis(gradient, -1, z)
    nchi = multiply(nz, dz/dchi)
    int1 = cumulative_trapezoid(nchi, chi, initial=0)
    int2 = cumulative_trapezoid(nchi/chi, chi, initial=0)
    gchi = (int1[...,-1,newaxis] - int1) - chi*(int2[...,-1,newaxis] - int2)
    return chi, gchi

def gchi_to_nz(chi, gchi, cosmo=cosmology.Planck15, niter=3):
    from numpy import min, max, linspace, interp, gradient, apply_along_axis
    from astropy.cosmology import z_at_value
    chiu = cosmo.comoving_distance(0).unit
    zmin = z_at_value(cosmo.comoving_distance, min(chi)*chiu)
    zmax = z_at_value(cosmo.comoving_distance, max(chi)*chiu)
    zint = linspace(zmin, zmax, 10*len(chi))
    for i in range(niter):
        cint = cosmo.comoving_distance(zint).value
        zint = interp(chi, cint, zint)
    z = zint
    dz = gradient(z)
    dchi = gradient(chi)
    dif1 = apply_along_axis(gradient, -1, gchi)
    dif2 = apply_along_axis(gradient, -1, dif1/dchi)
    nz = chi*dif2/dz
    return z, nz