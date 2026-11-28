"""
This module loads data from the powtable files which summarize the simulation
results made by Hunjing Huang et al.

This is justified from OWLS module in cosmosis standard library.

I interpolate into that data using a bivariate spline to get an estimate of the
effect on the matter power from baryons at a given z and k.

I could have made mistakes here easily in understanding what is happening! 

"""
from astropy.table import Table
from builtins import range
from builtins import object
import numpy as np
import re
import scipy.interpolate

import os
dirname = os.path.split(__file__)[0]

# File names
        
class RatioDatPowerModulator(object):
    """ using hunjing's logpk ratio fits file. redshift list is specific to her file"""

    def __init__(self, ratio_file):
        # Load the first line of the ratio file, which lists the
        # column names, each of which describes a different z value
        f = open(ratio_file, 'r')
        line = f.readline()
        f.close()
        colnames = line.split()[1:]

        # Assuming read-in redshift list is in decending order
        colnames.reverse()

        # Parse the column names to determine the redshifts they refer to.
        # They are in the form "logPk197" -> z = 1.97
        nz = len(colnames)
        redshifts = np.zeros(nz)
        for i,col in enumerate(colnames):
            # Pull out the integer part
            zintstr = re.findall(r'\d+',col)[0]
            # assume decimal place is after first digit
            redshifts[i]  = int(zintstr)/10.0**(len(zintstr)-1)

        print("Reading baryonic effects table {}, which has {} redshift values in".format(ratio_file, nz))

        data = np.loadtxt(ratio_file,skiprows=1)
        logk = data[:,0]

        logpkratio = np.zeros((len(logk),nz))

        # Again, we need to reverse the ordering of the file to put it
        # in redshift order
        for i in range(nz):
            logpkratio[:,i] = data[:,-(i+1)]

        # Make the spline that scales everything
        self.ratio = scipy.interpolate.RectBivariateSpline(logk, redshifts, logpkratio)
        self.ratio_k = logk

    def modulate(self, k, z, P, r=None):
        # They are definitely logs to base 10, because if you look
        # at the file 10**log_k_max is a nice round number
        logk = np.log10(k)

        # Compute the DM -> scenario scaling factor
        s = self.ratio(logk,z)
        modulation = 10**s

        # and scale and return the power
        P_out = P*modulation
        return P_out

