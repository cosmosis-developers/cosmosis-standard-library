import numpy as np
import sacc
import sys

# Set up the covariance matrix for the correlated Delta z prior

# Y1 fiducial
# 1) Which mock will you be analysing? 
intag = "Y10_variant"
# 2) What is the uncertainty in the bias correction?
delta_z = 0.001 # mimicing the z-dependent accuracy of the Y10 SRD
# 3) Do you want the uncertainty to have a z-dependence? 
# # if var_z = True, uncertainty in delta z bias correction is delta_z*(1+z_i), where z_i is the mean of tomographic bin i
var_z = True  
# 4) Do you want to include a correlation between bins?
correlation = True
# 5) What do you want the output file tagged as
outtag = 'Y10_with_Y10_SRD'

'''
# Y1 correlated
intag = "Y1_with_outliers"
delta_z = 0.002 # mimicing the z-dependent accuracy of SOM
var_z = True  
# if var_z = True, uncertainty in delta z bias correction is delta_z*(1+z_i), where z_i is the mean of tomographic bin i
correlation = True
outtag = 'Y1'

# Y1 uncorrelated
intag = "Y1_with_outliers"
delta_z = 0.002 # mimicing the z-dependent accuracy of SOM
var_z = True  
# if var_z = True, uncertainty in delta z bias correction is delta_z*(1+z_i), where z_i is the mean of tomographic bin i
correlation = False
outtag = 'Y1'

# Y1 without redshift outliers
intag = "Y1_no_z_outliers"
delta_z = 0.002 # mimicing the z-dependent accuracy of SOM
var_z = True  
# if var_z = True, uncertainty in delta z bias correction is delta_z*(1+z_i), where z_i is the mean of tomographic bin i
correlation = True
outtag = 'Y1_no_z_outliers'

# Y10 expected
tag = "Y10_variant"
delta_z = 0.015 # mimicing the z-independent accuracy of cross-correlation calibration
var_z = False
# if var_z = False, uncertainty in delta z bias correction has no z-dependence
correlation = True
outtag = 'Y10'

# Y10 optimistic
intag = "Y10_variant"
delta_z = 0.002 # mimicing the z-dependent accuracy of SOM, pretending we will have a deep spectroscopic sample
var_z = True
correlation = True
outtag = 'Y10-optimistic'

# Y1 with Y10 SRD
intag = "Y1_with_outliers"
delta_z = 0.001 # mimicing the Y10 SRD requirement, pretending we will have a deep spectroscopic sample
var_z = True
correlation = True
outtag = 'Y1_with_Y10_SRD'

# Y10 with Y10 SRD
intag = "Y10_variant"
delta_z = 0.001 # mimicing the Y10 SRD requirement,, pretending we will have a deep spectroscopic sample
var_z = True
correlation = True
outtag = 'Y10_with_Y10_SRD'

'''




num_z_bins = 5

#First we set the correlation between bins
#We will assume 20% correlation between neighbouring bins and a floor of 5% correlation across the rest
if correlation == True:
    corr_z = ([1, 0.2, 0.05, 0.05, 0.05],\
              [0.2, 1, 0.2, 0.05, 0.05],\
              [0.05, 0.2, 1, 0.2, 0.05],\
              [0.05, 0.05, 0.2, 1, 0.2],\
              [0.05, 0.05, 0.05, 0.2, 1])
else:
    corr_z = np.identity(num_z_bins)
                
# Now we ask if the error has a z-dependence
z_mean = np.zeros(num_z_bins)   #if not, this array will be filled with zeros
if var_z == True:
    #if yes, read in mock file to determine the mean z of each tomographic bin
    filename = '../mocks/mock-%s.fits'%(intag)
    mock = sacc.Sacc.load_fits(filename)

    #read in redshift distributions
    for tracer in mock.tracers.values():
        idx = int(tracer.name.split("_")[-1])
        z = tracer.z
        nz = tracer.nz
        z_mean[idx] = np.sum(z*nz)/np.sum(nz)

# Now we can construct the covariance matrix
sigma_z = delta_z * (np.ones(num_z_bins)+ z_mean)
cov_z = np.multiply(np.outer(sigma_z, sigma_z), corr_z)

if correlation == True:
    fileout = open("Delta_z_covariance_%s.txt"%(outtag), "w")
else:   
    fileout = open("uncorrelated_Delta_z_covariance_%s.txt"%(outtag), "w")    

np.savetxt(fileout, cov_z, delimiter = " ", fmt = "%1.4e")
fileout.close()

