import numpy as np
import math
import twopoint
from cosmosis.datablock import BlockError
from twopoint_cosmosis import type_table
from scipy.interpolate import interp1d
#We need to use the legendre module in 
#cosmosis-standard-library/shear/cl_to_xi_fullsky
import sys
import os
dirname = os.path.split(__file__)[0]
fullsky_path = os.path.join(dirname,"..","..","shear","cl_to_xi_fullsky")
sys.path.append(fullsky_path)
import legendre
from collections import OrderedDict

CL2XI_TYPES=["00","02+","22+","22-"]

"""
Class for interpolating spectra
"""
class SpectrumInterp(object):
    def __init__(self, angle, spec, bounds_error=False):
        assert np.all(angle>=0)
        #Check if the angle array starts with zero - 
        #this will affect how we interpolate
        starts_with_zero=False
        self.spec0 = 0.

        if angle[0]<1.e-9:
            #if the first angle value is 0,
            #record the first spec value as self.spec0
            #and then set angle and spec arrays to 
            #skip the first element.
            starts_with_zero=True
            self.spec0 = spec[0]
            angle, spec = angle[1:], spec[1:]

        if np.all(spec > 0):
            self.interp_func = interp1d(np.log(angle), np.log(
                spec), bounds_error=bounds_error, fill_value=-np.inf)
            self.interp_type = 'loglog'
            self.x_func = np.log
            self.y_func = np.exp

        elif np.all(spec < 0):
            self.interp_func = interp1d(
                np.log(angle), np.log(-spec), bounds_error=bounds_error, fill_value=-np.inf)
            self.interp_type = 'minus_loglog'
            self.x_func = np.log
            self.y_func = lambda y: -np.exp(y)
        else:
            self.interp_func = interp1d(
                np.log(angle), spec, bounds_error=bounds_error, fill_value=0.)
            self.interp_type = "log_ang"
            self.x_func = np.log
            self.y_func = lambda y: y

    def __call__(self, angle):
        non_zero = angle>1.e-12
        interp_vals = self.x_func(angle)
        try:
            spec = self.y_func( self.interp_func(interp_vals) )
        except ValueError:
            interp_vals[0] *= 1+1.e-9
            interp_vals[-1] *= 1-1.e-9
            spec = self.y_func( self.interp_func(interp_vals) )
        return np.where(non_zero, spec, self.spec0)

class TheorySpectrum(object):
    """Tomographic theory spectrum
    noise_var is a the noise variance per mode for each redshift bin
    """
    def __init__(self, name, types, nbin_a, nbin_b, angles, spectra, is_auto,
        noise_var_per_mode=None, auto_only = False,  bin_pairs=None):

        self.name = name
        self.types = types
        self.angles = angles
        self.is_auto = is_auto
        self.nbin_a = nbin_a
        self.nbin_b = nbin_b
        self.auto_only = auto_only
        self.spectra = spectra

        if self.auto_only and not self.is_auto:
            raise ValueError("A spectrum cannot be specified as auto_only without is_auto")

        self._set_bin_pairs(bin_pairs)

        self.noise_var_per_mode = noise_var_per_mode

        for bin_pair in self.bin_pairs:
            if bin_pair not in spectra:
                raise ValueError("Missing spectrum pairs {}".format(bin_pair))

    def set_noise(self, noise):
        self.noise_var_per_mode = noise

    def _set_bin_pairs(self, bin_pairs):
        if bin_pairs is not None:
            self.bin_pairs = bin_pairs
            return

        self.bin_pairs=[]
        for i in range(1, self.nbin_a+1):
            if self.is_auto:
                j_start = i
            else:
                j_start = 1
            for j in range(j_start, self.nbin_b+1):
                if self.auto_only:
                    if j!=i: continue
                self.bin_pairs.append((i,j))

    def cut_bin_pair(self, bin_pair):
        self.bin_pairs.remove(bin_pair)
        self.spec_interps.pop(bin_pair)

    @classmethod
    def from_block(cls, block, section_name, auto_only=False, bin_pairs=None):
        """
        Reads the predictions from the corresponding block and 
        initializes the Spectrum Measurement Class with some spectra.
        section_name: is either e.g. shear_cl or shear_cl_<save_name>
        auto_only: earlier in the pipeline, e.g. in project_2d.py, 
                  we may only have calculated auto-correlations 
                  for one of the spectra, to speed things up.  
                  TheorySpectrum.from_block however will currently attempt to read in all 
                  the correlations one would expect based on the number of redshift bins, 
                  unless you tell it not to, by having auto_only=True.
                  This option is used when called from save_2pt.py
        bin_pairs: Alternative option to auto_only, used when the method is called
                  from 2pt_like.py instead of from save_2pt. For auto-correlations will be
                  a list of tuples e.g. [(1,1), (2,2), ..., (n, n)]. If bin_pairs is provided, 
                  will only attempt to read those bin_pairs from the block.
        """

        spectrum_name = section_name
        save_name = block.get_string(section_name, "save_name")
        if save_name:
            spectrum_name = section_name.replace("_%s"%save_name, "")

        #Get types from spectrum_name
        type_table_items = list(type_table.items())
        type_names_list, type_info_list = [t[0] for t in type_table_items], [t[1] for t in type_table_items]
        type_index = [t[0] for t in type_info_list].index(spectrum_name)
        type_names, type_info = type_names_list[type_index], type_info_list[type_index]

        bin_format = type_info[-1]
        types = (  getattr(twopoint.Types, type_names[0]), 
                   getattr(twopoint.Types, type_names[1])  )

        # for cross correlations we must save bin_ji as well as bin_ij.
        # but not for auto-correlations. Also the numbers of bins can be different
        is_auto = block.get_bool( section_name, "is_auto" )

        if block.has_value(section_name, "nbin_a"):
            nbin_a = block[section_name, "nbin_a"]
            nbin_b = block[section_name, "nbin_b"]
        else:
            nbin_a = block[section_name, "nbin"]
            nbin_b = block[section_name, "nbin"]

        angle_name = block[section_name, "sep_name"]
        angles = block[section_name, angle_name]

        # If we have not been told in advance, decide which bin pairs
        # to use.
        if bin_pairs is None:
            bin_pairs = []
            for i in range(1, nbin_a + 1):
                j_start = i if is_auto else 1
                for j in range(j_start, nbin_b+1):
                    if (not auto_only) or (j == i):
                        bin_pairs.append((i,j))


        # now load from the block.
        spectra = OrderedDict()
        for (i,j) in bin_pairs:
            bin_name = bin_format.format(i, j)
            if is_auto and not block.has_value(section_name, bin_name):
                bin_name = bin_format.format(j, i)

            spectra[(i, j)] = block[section_name, bin_name]

        is_bin_averaged = block.get_bool(section_name, 'bin_avg', default=False)

        if is_bin_averaged:
            angle_edges = block[section_name, angle_name + "_edges"]

            return BinAveragedTheorySpectrum(
                spectrum_name, types, nbin_a, nbin_b, angles, spectra, is_auto, angle_edges,
                auto_only=auto_only, bin_pairs=bin_pairs)
        else:
            return InterpolatedTheorySpectrum(
                spectrum_name, types, nbin_a, nbin_b, angles, spectra, is_auto,
                auto_only=auto_only, bin_pairs=bin_pairs)

    def to_twopoint_object(self, angle_vals_out, kernels, output_name,
                                  angle_units='arcmin', windows="SAMPLE", angle_lims=None):

        # The fits format stores all the measurements
        # as one long vector.  So we build that up here from the various
        # bins that we will load in.  These are the different columns
        # angle_vals_out: angular values the user wants the spectra to be evaluated at. 
        # angle_vals_out can be different or the same as the ones from block, stored in self.angle_vals.

        bin1 = []
        bin2 = []
        value = []
        angular_bin = []
        angle = []
        angle_min = []
        angle_max = []
        n_angle_sample = len(angle_vals_out)

        def to_rad(x):
            if x is None:
                return x
            if angle_units is None:
                return x
            y = x * twopoint.ANGULAR_UNITS[angle_units]
            z = y.to(twopoint.ANGULAR_UNITS["rad"]).value
            return z

        angle_vals_out_rad = to_rad(angle_vals_out)
        angle_lims_rad = to_rad(angle_lims)


        # Find the angles we will interpolate at                    
        if self.is_bin_averaged:
            if angle_lims is None:
                raise ValueError("For now you must specify angle limits when saving bin-averaged spectra")
            angles = list(zip(angle_lims_rad[:-1], angle_lims_rad[1:]))
        else:
            angles = angle_vals_out_rad



        # Bin pairs. Varies depending on auto-correlation
        for (i,j) in self.bin_pairs:
            spec_sample = [self.get_spectrum_value(i, j, a)[0] for a in angles]

            # Build up on the various vectors that we need
            bin1.append(np.repeat(i, n_angle_sample))
            bin2.append(np.repeat(j, n_angle_sample))
            value.append(spec_sample)
            angular_bin.append(np.arange(n_angle_sample))
            angle.append(angle_vals_out)

            if angle_lims is not None:
                angle_min.append(angle_lims[:-1])
                angle_max.append(angle_lims[1:])


        # Convert all the lists of vectors into long single vectors
        bin1 = np.concatenate(bin1)
        bin2 = np.concatenate(bin2)
        angular_bin = np.concatenate(angular_bin)
        value = np.concatenate(value)
        angle = np.concatenate(angle)
        if angle_lims is not None:
            angle_min = np.concatenate(angle_min)
            angle_max = np.concatenate(angle_max)
        bins = (bin1, bin2)

        # Build the output object type reqired.
        s = twopoint.SpectrumMeasurement(output_name, bins, self.types, kernels, windows,
                                         angular_bin, value, angle=angle, 
                                         angle_unit=angle_units,
                                         angle_min=angle_min, angle_max=angle_max)
        return s

    def bin_pair_from_bin1_bin2( self, bin1, bin2):
        bin_pair = (bin1, bin2)
        if self.is_auto and (bin_pair not in self.bin_pairs):
            bin_pair = (bin2, bin1)
        return bin_pair


class BinAveragedTheorySpectrum(TheorySpectrum):
    is_bin_averaged = True
    def __init__(self, name, types, nbin_a, nbin_b, angles, spectra, is_auto, angle_edges,
        noise_var_per_mode=None, auto_only = False,  bin_pairs=None, ):

        super(BinAveragedTheorySpectrum, self).__init__(
            name, types, nbin_a, nbin_b, angles, spectra, is_auto,
            noise_var_per_mode=noise_var_per_mode, auto_only = auto_only,  bin_pairs=bin_pairs)

        self.angle_edges = angle_edges
        self.angle_mins = self.angle_edges[:-1]
        self.angle_maxs = self.angle_edges[1:]

    def get_noise_spec_values( self, bin1, bin2, angle ):
        raise NotImplementedError("get_noise_spec_values is not implemented for bin-averaged theory")

    def get_spectrum_value(self, bin1, bin2, angle):
        """
        Function that returns the value for the model at some bins and 
        particular angular scale. You can choose whether to interpolate the 
        spectra in the block or not. 
        bin1: redshift bin sample 1
        bin2: redshift bin sample 2
        angle: angular value in radians we want our corresponding model.
        angle_min: theta_edge min in radians for this bin.
        angle_min: theta_edge max in radians for this bin.
        """

        assert len(angle) == 2, "Code is mis-using BinAveragedTheorySpectrum - please complain to Joe"
        angle_min, angle_max = angle

        i, j = self.bin_pair_from_bin1_bin2(bin1, bin2)

        # Find correct index

        bin_index = np.argmin(abs(angle_min - self.angle_mins))
        if not (math.isclose(self.angle_mins[bin_index], angle_min)
                and math.isclose(self.angle_maxs[bin_index], angle_max)):
            raise ValueError("Desired angle bin {} - {} not computed".format(angle_min,angle_max))

        # Get corresponding spectra and angular mean value from block for this angular scale
        spectrum_value = self.spectra[(i,j)][bin_index]
        mid_angle = self.angles[bin_index]


        # Also return the angle valuues
        return spectrum_value, mid_angle


class InterpolatedTheorySpectrum(TheorySpectrum):
    is_bin_averaged = False


    def __init__(self, name, types, nbin_a, nbin_b, angles, spectra, is_auto,
        noise_var_per_mode=None, auto_only=False,  bin_pairs=None):

        super(InterpolatedTheorySpectrum, self).__init__(
            name, types, nbin_a, nbin_b, angles, spectra, is_auto,
            noise_var_per_mode=noise_var_per_mode, auto_only=auto_only,  bin_pairs=bin_pairs)

        self.splines = {}

    def get_spline(self, i, j):
        # Creating spline if it doesn't exist already
        if (i, j) not in self.splines:
            self.splines[(i,j)] = SpectrumInterp(self.angles, self.spectra[(i,j)])

        return self.splines[(i,j)]


    def _get_2point_vector_chunk(self, i, j, angles, angle_lims):
        return [self.get_spectrum_at_angle(angle) for angle in angles]



    def get_spectrum_value(self, bin1, bin2, angle):

        assert np.isscalar(angle), "Code is mis-using InterpolatedTheorySpectrum - please complain to Joe"

        i, j = self.bin_pair_from_bin1_bin2(bin1, bin2)
        spline = self.get_spline(i, j)

        try:
            spec_sample = spline(angle)
        except ValueError:
            raise ValueError("""
                Tried to get theory prediction for {} {}, but ell or theta value ({}) was out of range.
                "Maybe increase the range when computing/projecting or check units?""".format(
                    self.name, (bin1, bin2), angle))

        return spec_sample, spline




    def get_noise_spec_values( self, bin1, bin2, angle ):
        noise = np.zeros_like(angle, dtype=float)
        if self.is_auto and bin1 == bin2:
            n = self.noise_var_per_mode[bin1 - 1]
            noise = n(angle) if callable(n) else n * np.ones_like(angle)
        return noise

    def get_obs_spec_values( self, bin1, bin2, angle ):
        spec_vals = np.array([self.get_spectrum_value(bin1, bin2, a)[0] for a in angle])
        noise = self.get_noise_spec_values( bin1, bin2, angle )
        return spec_vals + noise


def cov2corr(cov):
    corr = np.zeros_like(cov)
    for i in range(cov.shape[0]):
        for j in range(cov.shape[1]):
            corr[i,j] = cov[i,j]/np.sqrt(cov[i,i]*cov[j,j])
    return corr

def arcmin_to_rad(angle):
    return np.radians(angle/60.)

def convert_angle(angle, orig_unit, new_unit):
    #Convert angular units to arcmin
    angle_with_units = angle * orig_unit
    return (angle_with_units.to(new_unit)).value

def perarcmin2_to_perrad2(n):
    n = np.array(n)
    arcmin2_per_rad2 = np.degrees( np.degrees( 1. ) ) * 60 * 60
    return n * arcmin2_per_rad2

def get_types(cl_name):

    #Get types from spectrum_name
    for type_names, type_info in type_table.items():
        if cl_name == type_info[0]:
            break
    bin_format = type_info[-1]
    types = (  getattr(twopoint.Types, type_names[0]), 
               getattr(twopoint.Types, type_names[1])  )
    return types

def downsample_block( angle_lims_orig, angle_mids_orig, cov_orig, n_out ):
    #make sure cov length is divisble by n_out
    #weight by dtheta*angle_mid (weighting by n_pairs geometric expectation)
    dtheta = angle_lims_orig[1:]-angle_lims_orig[:-1]
    assert cov_orig.shape[0]%n_out == 0
    norig_per_nout = cov_orig.shape[0]//n_out
    cov_out = np.zeros( (n_out, n_out) )
    angle_mids_out = np.zeros( n_out )
    weights = angle_mids_orig * dtheta
    for i in range(n_out):
        orig_inds_i = np.arange(i*norig_per_nout, (i+1)*norig_per_nout)
        sum_w_i = (weights[orig_inds_i]).sum()
        angle_mids_out[i] = (weights[orig_inds_i] * angle_mids_orig[orig_inds_i]).sum() / sum_w_i
        for j in range(n_out):
            orig_inds_j = np.arange(j*norig_per_nout, (j+1)*norig_per_nout)
            sum_w_j = (weights[orig_inds_j]).sum()
            cov_orig_inds_ij = np.ix_(orig_inds_i, orig_inds_j)
            cov_out[i,j] = np.matmul( weights[orig_inds_i], np.matmul( cov_orig[cov_orig_inds_ij], weights[orig_inds_j] ) )/sum_w_i/sum_w_j
    return cov_out, angle_mids_out

class ClCov( object ):
    """
    Class for computing cl covariance
    """
    def __init__(self, theory_spectra, fsky=1.):
        self.theory_spectra = theory_spectra
        self.types = [ t.types for t in self.theory_spectra ]
        self.names = [t.name for t in self.theory_spectra ]
        self.fsky = fsky

    def get_cov_diag_ijkl( self, name1, name2, ij, kl, ell_max, ell_min=0, noise_only=False):
        # From Joachimi & Bridle 2010 0911.2454
        # Cov(C^{ij}_{12}, C^{kl}_{34}) = prefactor * [ C^{ik}_{13} C^{jl}_{24} + C^{il}_{14} C^{kl}_{23} ]
        # The C's on the RHS are observed C(l)s (i.e. with noise bias)
        # superscripts indicate redshift bin pairs, subscripts quantities (shear or galaxy over-density etc.)
        ell_vals = np.arange(ell_min, ell_max+1)
        n_ell = len(ell_vals)

        c_ij_12 = self.theory_spectra[ self.names.index(name1) ]
        c_kl_34 = self.theory_spectra[ self.names.index(name2) ]

        cl2_sum = self.get_cl2sum_ijkl( c_ij_12, c_kl_34, ij, kl, ell_vals, 
            noise_only=noise_only )
        n_modes = self.fsky * (2*ell_vals+1)
        cov_diag  = ( cl2_sum ) / n_modes  
        return cov_diag  

    def get_cl2sum_ijkl( self, c_ij_12, c_kl_34, ij, kl, ells, noise_only=False):
        #Get the spectra C^{ik}_{13}, C^{jl}_{24}, C^{il}_{14}, C^{kl}_{23}
        #t1 and t2 are the two spectra C^{ij}_{12} and C^{kl}_{34}
        #So constuct the types for the desired spectra from these type pairs:
        i,j = ij
        k,l = kl

        type_1, type_2 = c_ij_12.types
        type_3, type_4 = c_kl_34.types

        bin_pairs = [ (i,k), (j,l), (i,l), (j,k) ]
        type_pairs = [ (type_1,type_3), (type_2,type_4), (type_1,type_4), (type_2,type_3) ]
        cl2_sum = np.zeros_like(ells, dtype=float)
        c_ells = []
        for bin_pair,type_pair in zip(bin_pairs, type_pairs):
            bin1, bin2 = bin_pair
            t1, t2 = type_pair
            types = (t1, t2)
            if types not in self.types:
                #If we don't have a spectra with these types, we probably 
                #have an equivalent one with the types the other way round.
                #In this case, we also need to swap bin1 and bin2, because
                #we are accessing e.g. C^{ik}_{13} via C^{ki}_{31}.
                types = (types[1], types[0])
                bin1, bin2 = bin2, bin1
            s = self.theory_spectra[ self.types.index( types ) ]
            if noise_only:
                c_ells.append(s.get_noise_spec_values( bin1, bin2, ells ))
            else:
                c_ells.append(s.get_obs_spec_values( bin1, bin2, ells ))
        #print(ret)
        return c_ells[0]*c_ells[1] + c_ells[2]*c_ells[3]

    def get_binned_cl_cov( self, ell_lims, noise_only=False):
        #-Build a binned C(l) covariance
        #-Assume the measured C(l) for bin i is the (2l+1)-weighted average 
        #of C(ell_lims[i],...,ell_lims[i+1]-1)
        #First construct the output array
        n_spectra = len(self.theory_spectra)
        n_ell = len(ell_lims) - 1
        n_dv = 0
        cl_lengths = []
        for s in self.theory_spectra:
            l = n_ell * len(s.bin_pairs)
            n_dv += l
            cl_lengths.append(l)
        covmat = np.zeros((n_dv, n_dv))
        ell_max = ell_lims[-1]
        
        #Get the starting index in the full datavector for each spectrum
        #this will be used later for adding covariance blocks to the full matrix.
        cl_starts = []
        start = 0
        for i in range(n_spectra):
            cl_starts.append( int(sum(cl_lengths[:i])) )

        #Now loop through pairs of Cls and pairs of bin pairs filling the covariance matrix
        for i_cl in range(n_spectra):
            cl_spec_i = self.theory_spectra[i_cl]
            for j_cl in range(i_cl, n_spectra):
                cl_spec_j = self.theory_spectra[j_cl]
                cov_blocks = {} #collect cov_blocks in this dictionary
                for i_bp, bin_pair_i in enumerate(cl_spec_i.bin_pairs):
                    for j_bp, bin_pair_j in enumerate(cl_spec_j.bin_pairs):
                        print(f"Computing covariance {i_cl},{j_cl} pairs <{bin_pair_i} {bin_pair_j}>")
                        #First check if we've already calculated this
                        if (i_cl == j_cl) and cl_spec_i.is_auto and ( j_bp < i_bp ):
                            cl_var_binned = cov_blocks[j_bp, i_bp]
                        else:
                            #First calculate the unbinned Cl covariance
                            ell_max = ell_lims[-1]
                            cl_var_unbinned = self.get_cov_diag_ijkl( cl_spec_i.name, 
                                cl_spec_j.name, bin_pair_i, bin_pair_j, ell_max, 
                                ell_min=ell_lims[0], noise_only=noise_only )
                            #Now bin this diaginal covariance
                            #Var(binned_cl) = \sum_l Var(w_l^2 C(l)) / (\sum_l w_l)^2
                            #where w_l = 2*l+1
                            cl_var_binned = np.zeros(n_ell)
                            for ell_bin, (ell_low, ell_high) in enumerate(zip(ell_lims[:-1], ell_lims[1:])):
                                #Get the ell values for this bin:
                                ell_vals_bin = np.arange(ell_low, ell_high).astype(int)
                                #Get the indices in cl_var_binned these correspond to:
                                ell_vals_bin_inds = ell_vals_bin - ell_lims[0]
                                cl_var_unbinned_bin = cl_var_unbinned[ell_vals_bin_inds]
                                cl_var_binned[ell_bin] = np.sum((2*ell_vals_bin+1)**2 * 
                                    cl_var_unbinned_bin) / np.sum(2*ell_vals_bin+1)**2

                            cov_blocks[i_bp, j_bp] = cl_var_binned

                        #Now work out where this goes in the full covariance matrix
                        #and add it there.
                        inds_i = np.arange( cl_starts[i_cl] + n_ell*i_bp, 
                            cl_starts[i_cl] + n_ell*(i_bp+1) )
                        inds_j = np.arange( cl_starts[j_cl] + n_ell*j_bp, 
                            cl_starts[j_cl] + n_ell*(j_bp+1) )
                        cov_inds = np.ix_( inds_i, inds_j )
                        covmat[ cov_inds ] = np.diag(cl_var_binned)
                        cov_inds_T = np.ix_( inds_j, inds_i )
                        covmat[ cov_inds_T ] = np.diag(cl_var_binned)

        print("Completed covariance")
        print("   Signed log det:", np.linalg.slogdet(covmat))
        print("   Condition number:", np.linalg.cond(covmat))
        return covmat, cl_lengths


def real_space_cov( cl_cov, cl_specs, cl2xi_types, ell_max, angle_lims_rad, 
    upsample=None, high_l_filter=0.75, noise_only=False ):
    """
    Compute real space covariance given cl covariance
    Add cov blocks to a dictionary with keys: spec_index_i spec_index_j binpair_index_i binpair_index_j
    """
    cov_blocks = {}
    ntheta = len(angle_lims_rad) - 1
    n_dv = 0
    xi_starts = []
    xi_lengths = []

    log_angle_lims_rad = np.log( angle_lims_rad )
    log_angle_mids_rad = 0.5*(log_angle_lims_rad[:-1]+log_angle_lims_rad[1:])
    angle_mids_rad = np.exp( log_angle_mids_rad )
    angle_lims_rad = np.exp( log_angle_lims_rad )
    dangle = angle_lims_rad[1:]-angle_lims_rad[:-1]
    if upsample is not None:
        log_angle_lims_rad_upsampled = np.linspace( log_angle_lims_rad[0], log_angle_lims_rad[-1], ntheta*upsample+1 )
        log_angle_mids_rad_upsampled = 0.5 * ( log_angle_lims_rad_upsampled[:-1] + log_angle_lims_rad_upsampled[1:] )
        angle_lims_rad_upsampled = np.exp( log_angle_lims_rad_upsampled )
        angle_mids_rad_upsampled = np.exp( log_angle_mids_rad_upsampled )
    else:
        angle_lims_rad_upsampled = angle_lims_rad
        angle_mids_rad_upsampled = np.exp( log_angle_mids_rad )

    # Identify spectrum types that carry ell-dependent noise (e.g. CMB lensing).
    # For these, the analytic noise-noise term is not valid and we use the full
    # numerical transform instead.
    l_dep_noise_types = set()
    for spec in cl_cov.theory_spectra:
        if spec.noise_var_per_mode is not None:
            if any(callable(n) for n in spec.noise_var_per_mode):
                l_dep_noise_types.update(spec.types)

    for i_xi in range(len(cl2xi_types)):
        cl2xi_i = cl2xi_types[i_xi]
        cl_spec_i = cl_specs[i_xi]

        F_i_l = legendre.get_F_theta_l(angle_mids_rad_upsampled, ell_max, cl2xi_i, high_l_filter=high_l_filter)
        #Record datavector lengths
        xi_starts.append(n_dv)
        n_dv += len(cl_spec_i.bin_pairs)*ntheta
        xi_lengths.append( n_dv - xi_starts[i_xi] )

        for j_xi in range(i_xi, len(cl2xi_types)):
            cl2xi_j = cl2xi_types[j_xi]
            cl_spec_j = cl_specs[ j_xi ]
            if j_xi==i_xi:
                F_j_l = F_i_l
            else:
                F_j_l = legendre.get_F_theta_l(angle_mids_rad_upsampled, ell_max, cl2xi_j, high_l_filter=high_l_filter)
            #Now loop through bin pairs for each spectrum
            #The calculation for each pair of bin_pairs is done via matrix multiplication...maybe
            #we could speed things up by replacing the below for loops with more matrix multiplication,
            #although it may be that the slow part is calculating the Cls, for which we need 
            #for loops anyway...
            for i_bp, bin_pair_i in enumerate(cl_spec_i.bin_pairs):
                for j_bp, bin_pair_j in enumerate(cl_spec_j.bin_pairs):
                    print(i_xi, j_xi, bin_pair_i, bin_pair_j)
                    print(cl_spec_i.name, cl_spec_j.name)
                    #Check if we've already done this combo
                    #We've already done it if its an auto-correlation i.e. i_xi==j_xi and
                    #j_bp is less than i_bp 
                    if (i_xi == j_xi) and cl_spec_i.is_auto and ( j_bp < i_bp ):
                        xi_cov_block = cov_blocks[i_xi, j_xi, j_bp, i_bp]
                    else:
                        #Get the full cl covariance, and the pure noise part - we're going to transform
                        #the latter analytically...
                        cl_cov_block = cl_cov.get_cov_diag_ijkl( cl_spec_i.name, 
                            cl_spec_j.name, bin_pair_i, bin_pair_j, ell_max, 
                            noise_only=noise_only )
                        cl_cov_noise_noise = cl_cov.get_cov_diag_ijkl( cl_spec_i.name, 
                            cl_spec_j.name, bin_pair_i, bin_pair_j, ell_max, noise_only=True )
                        # Check if either spectrum has ell-dependent noise (e.g. CMB lensing).
                        # If so, skip the analytic noise term and transform everything numerically.
                        pair_types = set(cl_spec_i.types) | set(cl_spec_j.types)
                        use_analytic_noise = not bool(pair_types & l_dep_noise_types)
                        if use_analytic_noise:
                            cl_cov_block_signal_mixed = cl_cov_block - cl_cov_noise_noise
                        else:
                            cl_cov_block_signal_mixed = cl_cov_block
                        xi_cov_block_signal_mixed_upsampled = np.matmul( F_i_l,
                                                              np.matmul( np.diag(cl_cov_block_signal_mixed), 
                                                              F_j_l.T ) )
                        xi_cov_block_signal_mixed, angle_mids = downsample_block( angle_lims_rad_upsampled,
                            angle_mids_rad_upsampled, xi_cov_block_signal_mixed_upsampled, ntheta )
                        if use_analytic_noise:
                            #For the analytic calculation, we use the fact that
                            #\int l J_nu(l theta) J_nu(l theta) = 1/theta.
                            #The noise term is
                            #(1/4 pi^2) \int dtheta \int l J_nu(l theta) J_nu(l theta) * noise = noise / theta / 4pi^2
                            noise = cl_cov_noise_noise[0] #This is 2*cl_noise^2/fsky.
                            #if shear-shear, we need to multiply this noise by 2, since we have noise from both E and B-modes.
                            #Feel like a hack adding it at this stage...maybe there is a more motivated way...
                            if (cl2xi_i==cl2xi_j) and (cl2xi_i in ["22+","22-"]):
                                noise *= 2
                            else:
                                assert cl2xi_i in CL2XI_TYPES

                            #For gg, npairs = pi*theta*dtheta*n_gal^2*area = 4*pi^2*fsky*n_gal^2*theta*dtheta
                            # cl_noise = 1./n_gal^2, so npairs = 4 * pi^2 * theta * dtheta / (cl_noise^2 / fsky)
                            # = 8 * pi^2 * theta * dtheta / (2 * cl_noise^2 / fsky)
                            # = 8 * pi^2 * theta * dtheta / noise
                            xi_cov_block_noise_noise_diag = (noise/angle_mids_rad/(8*np.pi*np.pi)/dangle )
                            xi_cov_block = xi_cov_block_signal_mixed + np.diag(xi_cov_block_noise_noise_diag)
                        else:
                            xi_cov_block = xi_cov_block_signal_mixed
                    cov_blocks[i_xi, j_xi, i_bp, j_bp] = xi_cov_block

    #construct full covariance
    covmat = np.zeros((n_dv, n_dv))

    for block_key, block_vals in cov_blocks.items():
        #get datavector indices
        i_xi, j_xi, i_bp, j_bp = block_key
        inds_i = np.arange( xi_starts[i_xi] + ntheta*i_bp, xi_starts[i_xi] + ntheta*(i_bp+1) )
        inds_j = np.arange( xi_starts[j_xi] + ntheta*j_bp, xi_starts[j_xi] + ntheta*(j_bp+1) )
        cov_inds = np.ix_( inds_i, inds_j )
        covmat[ cov_inds ] = block_vals
        cov_inds_T = np.ix_( inds_j, inds_i )
        covmat[ cov_inds_T ] = block_vals.T

    print("Completed covariance")
    print("slog det:", np.linalg.slogdet(covmat))
    print("condition number:", np.linalg.cond(covmat))

    return cov_blocks, covmat, xi_starts, xi_lengths


def real_space_cov_contributions(cl_cov, cl_specs, cl2xi_types, ell_max, angle_lims_rad,
                                  upsample=None, high_l_filter=0.75):
    """
    Decompose the real-space covariance into contributions from individual noise sources.

    Returns a dict with keys:
      'signal'            : covariance with all noise zeroed (cosmic variance only)
      'shot_noise'        : contribution from galaxy shot noise
      'shape_noise'       : contribution from source shape noise
      'cmb_lensing_noise' : contribution from CMB lensing reconstruction noise

    Each noise contribution is defined as cov(signal + that_noise) - cov(signal_only),
    so it includes both the linear (signal x noise) and quadratic (noise x noise) terms
    for that noise source.  The sum of the four contributions may differ from the full
    covariance by small cross-noise terms (e.g. shot x shape), which are typically
    negligible compared to the dominant contributions.
    """
    import copy

    noise_type_labels = {
        "galaxy_position_fourier": "shot_noise",
        "galaxy_shear_emode_fourier": "shape_noise",
        "cmb_kappa_fourier": "cmb_lensing_noise",
    }

    def clone_cl_cov_keep(keep_type_names):
        """Return a ClCov with noise zeroed for all tracer types NOT in keep_type_names."""
        new_spectra = []
        for spec in cl_cov.theory_spectra:
            if (spec.noise_var_per_mode is not None
                    and spec.types[0].name not in keep_type_names):
                new_spec = copy.copy(spec)
                # Use zeros list (not None) to keep get_noise_spec_values indexing intact
                new_spec.noise_var_per_mode = [0.0] * len(spec.noise_var_per_mode)
                new_spectra.append(new_spec)
            else:
                new_spectra.append(spec)
        return ClCov(new_spectra, fsky=cl_cov.fsky)

    def run_cov(cl_cov_subset):
        _, covmat, _, _ = real_space_cov(
            cl_cov_subset, cl_specs, cl2xi_types,
            ell_max, angle_lims_rad,
            upsample=upsample, high_l_filter=high_l_filter)
        return covmat

    print("Computing signal-only covariance contribution...")
    cov_signal = run_cov(clone_cl_cov_keep([]))
    contributions = {"signal": cov_signal}

    for type_name, label in noise_type_labels.items():
        print(f"Computing {label} covariance contribution...")
        cov_with_noise = run_cov(clone_cl_cov_keep([type_name]))
        contributions[label] = cov_with_noise - cov_signal

    return contributions
