from astropy.io import fits
import astropy.units
from astropy.table import Table
from enum import Enum
import numpy as np
import copy
import os
import warnings


# FITS header keyword indicating 2-point data
TWOPOINT_SENTINEL = "2PTDATA"
NZ_SENTINEL = "NZDATA"
COV_SENTINEL = "COVDATA"
window_types = ["SAMPLE", "CLBP", "LOG_MID"]
# LOG_MID interpolates at log mid of bin

METADATA_PREFIX = "MD_"

# Please do not add things to this list
ANGULAR_UNIT_TYPES = [
    astropy.units.arcsec,
    astropy.units.arcmin,
    astropy.units.rad,
    astropy.units.deg,
]
ANGULAR_UNITS = {unit.name: unit for unit in ANGULAR_UNIT_TYPES}


def sample_cov(xi_arrays, mode='full'):
    """mode should be full, subsample or jk"""
    nsample, npoints = xi_arrays.shape
    Cov = np.zeros((npoints, npoints))
    Corr = np.zeros_like(Cov)

    xi_mean = np.mean(xi_arrays, axis=0)
    for i in range(npoints):
        for j in range(npoints):
            Cov[i, j] = np.sum((xi_arrays[:, i] - xi_mean[i])
                               * (xi_arrays[:, j] - xi_mean[j])) / nsample
    # This is the covariance in a patch of size A/nsample. Assume Cov ~ 1/A so Cov=Cov/nsample
    if mode == 'subsample':
        Cov /= nsample
    elif mode == "jk":
        Cov *= (nsample - 1)
    for i in range(npoints):
        for j in range(npoints):
            Corr[i, j] = Cov[i, j] / np.sqrt(Cov[i, i] * Cov[j, j])
    return Cov, Corr


class Types(Enum):
    """
    This is an enumeration - a list of possible values with names and code values
    that we can use in FITS headers.

    It enumerates the different quantities that a two-point measurement can correlate.
    For example, CMB T,E,B, lensing E and B, galaxy position and magnification.

    It specifies the quantity and whether we are in Fourier space or real space.

    One special case is xi_{-} and xi_{+} in galaxy shear.  These values are already
    correlations of combinations of E and B modes. We denote this as xi_{++} and xi_{--} here.

    """
    galaxy_position_fourier = "GPF"
    galaxy_shear_emode_fourier = "GEF"
    galaxy_shear_bmode_fourier = "GBF"
    galaxy_position_real = "GPR"
    galaxy_shear_plus_real = "G+R"
    galaxy_shear_minus_real = "G-R"
    cmb_kappa_real = "CKR"
    cmb_kappa_fourier = "CKF"
    cmb_kappa_emode_fourier = "CKF"
    
    @classmethod
    def lookup(cls, value):
        for T in cls:
            if T.value == value:
                return T


def dummy_kernel(name):
    return NumberDensity(name, np.zeros(10), np.zeros(10), np.zeros(10), [np.zeros(10)])


class NumberDensity(object):
    """
    This class contains n(z) information for a collection of numbered redshift bins.
    It is expected to be used for a single sample type (e.g. galaxy sample) split into
    tomographic bins rather than the more complex cases where there are multiple different
    quantities (e.g. two galaxy samples).

    Since the main use case for this is photometric redshifts, and photo-z codes typically
    produce histogram type data sets (that is, they look like step functions between each bin),
    that is what this form assumes.
    """

    def __init__(self, name, zlow, z, zhigh, nzs, ngal=None, sigma_e=None):
        self.name = name
        self.zlow = zlow
        self.z = z
        self.zhigh = zhigh
        self.nbin = len(nzs)
        if self.nbin > 0:
            self.nsample = len(nzs[0])
        else:
            self.nsample = 0
        self.nzs = nzs
        self.ngal = ngal
        self.sigma_e = sigma_e

    @classmethod
    def from_fits(cls, extension):
        # load in the n(z) data
        data = extension.data
        header = extension.header

        z = data['Z_MID']
        zlow = data['Z_LOW']
        zhigh = data['Z_HIGH']
        i = 1
        name = 'BIN{}'.format(i)
        nzs = []
        sigma_e = []
        ngal = []
        while name in data.names:
            nz = data[name]
            nzs.append(nz)
            ngal.append(header.get('NGAL_{}'.format(i)))
            sigma_e.append(header.get('SIG_E_{}'.format(i)))
            i += 1
            name = 'BIN{}'.format(i)

        if all(x is None for x in sigma_e):
            sigma_e = None
        else:
            assert not any(x is None for x in sigma_e), "Please specify all or none of the SIG_E_ in your n(z) section {}".format(
                extension.name)
            sigma_e = np.array(sigma_e)

        if all(x is None for x in ngal):
            ngal = None
        else:
            assert not any(x is None for x in ngal), "Please specify all or none of the NGAL in your n(z) section {}".format(
                extension.name)
            ngal = np.array(ngal)

        N = cls(extension.name, zlow, z, zhigh,
                nzs, ngal=ngal, sigma_e=sigma_e)

        return N

    def to_fits(self):
        header = fits.Header()
        header[NZ_SENTINEL] = True
        header['EXTNAME'] = self.name

        columns = [
            fits.Column(name='Z_LOW', array=self.zlow, format='D'),
            fits.Column(name='Z_MID', array=self.z, format='D'),
            fits.Column(name='Z_HIGH', array=self.zhigh, format='D'),
        ]

        for i in range(self.nbin):
            name = "BIN{}".format(i + 1)
            columns.append(fits.Column(
                name=name, array=self.nzs[i], format='D'))

        if self.sigma_e is not None:
            for i in range(self.nbin):
                name = "SIG_E_{}".format(i + 1)
                header[name] = self.sigma_e[i]

        if self.ngal is not None:
            for i in range(self.nbin):
                name = "NGAL_{}".format(i + 1)
                header[name] = self.ngal[i]

        extension = fits.BinTableHDU.from_columns(columns, header=header)
        return extension

    @classmethod
    def from_block(cls, block, section_name, output_name=None):
        """Load kernel from cosmosis datablock"""
        if output_name == None:
            output_name = section_name
        if block.has_value(section_name, "z_mid"):
            z_mid = block[section_name, "z_mid"]
            z_low = block[section_name, "z_low"]
            z_high = block[section_name, "z_high"]
        else:
            z_mid = block[section_name, "z"]
            dz = z_mid[2] - z_mid[1]
            z_low = z_mid - 0.5 * dz
            z_high = z_mid + 0.5 * dz

        nzs = []
        for i in range(1, 99999):
            bin_name = "bin_%d" % i
            if block.has_value(section_name, bin_name):
                nzs.append(block[section_name, bin_name])
            else:
                break

        N = cls(output_name, z_low, z_mid, z_high, nzs)
        return N


class SpectrumMeasurement(object):
    def __init__(self, name, bins, types, kernels, windows, angular_bin, value,
                 angle=None, error=None, angle_unit=None, metadata=None, npairs=None,
                 varxi=None, extra_cols=None, angle_min=None, angle_max=None):
        """
        Metadata is a dictionary which will get added to the fits header
        extra cols is a dictionary of tuples
        """
        self.name = name
        self.bin1, self.bin2 = bins
        self.bin_pairs = self.get_bin_pairs()  # unique bin pairs
        self.type1, self.type2 = types
        self.kernel1, self.kernel2 = kernels
        self.angular_bin = angular_bin
        self.angle = angle
        self.angle_min = angle_min
        self.angle_max = angle_max
        self.value = value
        self.npairs = npairs
        self.varxi = varxi
        if windows in window_types:
            self.windows = windows
        else:
            raise TypeError("window type %s not recognised" % windows)
        self.error = error
        self.metadata = metadata
        if self.is_real_space():
            #angle is real
            msg = "Files with real-space units must specify units as one of: {}".format(
                list(ANGULAR_UNITS.keys()))
            assert angle_unit in ANGULAR_UNITS,  msg
        self.angle_unit = angle_unit
        # This is the index in the datavector, will be populated when used in TwoPointFile
        self.dv_index = None
        self.extra_cols = extra_cols

    def __str__(self):
        return "<Spectrum: {}>".format(self.name)

    def __repr__(self):
        return "<Spectrum: {}>".format(self.name)

    def get_bin_pairs(self):
        unique_pairs = []
        for p in zip(self.bin1, self.bin2):
            if p not in unique_pairs:
                unique_pairs.append(p)
        return unique_pairs

    def canonical_order(self):
        # order by b1, then b2, then angle
        order = np.lexsort([self.bin1, self.bin2, self.angular_bin])
        return order

    def is_real_space(self):
        return self.type1.value.endswith("R") or self.type2.value.endswith("R")

    def recompute_angular_bins(self):
        angles = np.unique(self.angle)
        angles.sort()
        angles = angles.tolist()
        angular_bin = [angles.index(ang)+1 for ang in self.angle]
        self.angular_bin = np.array(angular_bin)

    def convert_angular_units(self, unit):
        if not self.is_real_space():
            raise ValueError(
                "Two point spectrum has no units to convert; it is in Fourier space")

        if self.windows not in ["SAMPLE", "CLBP"]:
            raise NotImplementedError(
                "Need to write code to transform window function units")
        old_unit = ANGULAR_UNITS[self.angle_unit]
        new_unit = ANGULAR_UNITS[unit]

        angle_with_units = self.angle * old_unit
        self.angle = angle_with_units.to(new_unit).value
        self.angle_unit = unit
        if self.angle_min is not None:
            self.angle_min = (self.angle_min * old_unit).to(new_unit).value
            self.angle_max = (self.angle_max * old_unit).to(new_unit).value

    def apply_mask(self, mask):
        """mask is a boolean array which True for elements to be kept"""
        self.bin1 = self.bin1[mask]
        self.bin2 = self.bin2[mask]
        self.angular_bin = self.angular_bin[mask]
        self.angle = self.angle[mask]
        self.value = self.value[mask]
        if self.error is not None:
            self.error = self.error[mask]
        if self.angle_min is not None:
            self.angle_min = self.angle_min[mask]
            self.angle_max = self.angle_max[mask]

    def cut_bin_pair(self, bin_pair, complain=False):
        """Cut a full bin pair. If complain is True,
        raise a ValueError if the bin pair is not found"""
        b1, b2 = bin_pair
        mask = (self.bin1 == b1) * (self.bin2 == b2)
        if (mask.sum() == 0) and complain:
            raise ValueError("""You tried to cut bin pair %d,%d from spectrum named %s 
                             but it doesn't exist""" % (b1, b2, self.name))
        elif mask.sum() > 0:
            self.apply_mask(~mask)

    def auto_bins(self):
        return self.bin1 == self.bin2

    def __len__(self):
        return len(self.value)

    def nbin(self):
        return np.max([self.bin1.max(), self.bin2.max()])

    def get_pair(self, bin1, bin2):
        w = (self.bin1 == bin1) & (self.bin2 == bin2)
        return self.angle[w], self.value[w]

    def get_pair_mask(self, bin1, bin2):
        w = (self.bin1 == bin1) & (self.bin2 == bin2)
        return w

    def get_error(self, bin1, bin2):
        if self.error is None:
            return None
        w = (self.bin1 == bin1) & (self.bin2 == bin2)
        return self.error[w]

    @classmethod
    def from_fits(cls, extension, covmat_info=None):
        name = extension.name
        # determine the type of the quantity involved
        type1 = Types.lookup(extension.header['QUANT1'])
        type2 = Types.lookup(extension.header['QUANT2'])

        # and the name of the kernels and the window function
        # extensions
        kernel1 = extension.header['KERNEL_1']
        kernel2 = extension.header['KERNEL_2']
        windows = extension.header['WINDOWS']

        if windows not in window_types:
            raise NotImplementedError(
                "Have not yet coded window functions for angular bins")

        # Now load the data
        data = extension.data
        bin1 = data['BIN1']
        bin2 = data['BIN2']
        angular_bin = data['ANGBIN']
        value = data['VALUE']
        if "ANG" in data.names:
            angle = data['ANG']
            ang_index = data.names.index("ANG")
            angle_unit = extension.header.get('TUNIT{}'.format(ang_index + 1))
            if angle_unit is not None:
                angle_unit = angle_unit.strip()
        else:
            angle = None
            angle_unit = None
        if "NPAIRS" in data.names:
            npairs = data['NPAIRS']
        else:
            npairs = None
        if "VARXI" in data.names:
            varxi = data['VARXI']
        else:
            varxi = None
        if "ANGLEMIN" in data.names:
            angle_min = data['ANGLEMIN']
        else:
            angle_min = None
        if "ANGLEMAX" in data.names:
            angle_max = data['ANGLEMAX']
        else:
            angle_max = None

        # check for extra columns
        found_extra_cols = False
        for c in data.names:
            if c.startswith("XTRA_"):
                if not found_extra_cols:
                    extra_cols = {}
                    found_extra_cols = True
                colname = (c.replace("XTRA_", "")).lower()
                extra_cols[colname] = data[c]
        if not found_extra_cols:
            extra_cols = None

        # check for metadata
        metadata = {}
        for key in extension.header:
            if key.startswith(METADATA_PREFIX):
                metadata[key.replace(METADATA_PREFIX, "")
                         ] = extension.header[key]

        # Load a chunk of the covariance matrix too if present.
        if covmat_info is None:
            error = None
        else:
            error = covmat_info.get_error(name)

        return SpectrumMeasurement(name, (bin1, bin2), (type1, type2), (kernel1, kernel2), windows,
                                   angular_bin, value, angle, error, angle_unit=angle_unit, npairs=npairs,
                                   varxi=varxi, angle_min=angle_min, angle_max=angle_max,
                                   extra_cols=extra_cols, metadata=metadata)

    def to_fits(self):
        header = fits.Header()
        header[TWOPOINT_SENTINEL] = True
        header['EXTNAME'] = self.name
        header['QUANT1'] = self.type1.value
        header['QUANT2'] = self.type2.value
        header['KERNEL_1'] = self.kernel1
        header['KERNEL_2'] = self.kernel2
        header['WINDOWS'] = self.windows  # NOT YET CODED ANYTHING ELSE
        header['N_ZBIN_1'] = len(np.unique(self.bin1))
        header['N_ZBIN_2'] = len(np.unique(self.bin2))
        if (self.metadata is not None) and self.metadata:
            # Check metadata doesn't share any keys with the stuff that's already in the header
            assert set(self.metadata.keys()).isdisjoint(set(header.keys()))
            for key, val in self.metadata.iteritems():
                # Use MD_ prefix so metadata entries can be easily recognised by from_fits
                header[METADATA_PREFIX+key] = val
        header['N_ANG'] = len(np.unique(self.angular_bin))

        columns = [
            fits.Column(name='BIN1', array=self.bin1, format='K'),
            fits.Column(name='BIN2', array=self.bin2, format='K'),
            fits.Column(name='ANGBIN', array=self.angular_bin, format='K'),
            fits.Column(name='VALUE', array=self.value, format='D'),
        ]

        if self.angle_min is not None:
            columns.append(fits.Column(
                    name='ANGLEMIN', array=self.angle_min, format='D', unit=self.angle_unit))
        if self.angle_max is not None:
            columns.append(fits.Column(
                    name='ANGLEMAX', array=self.angle_max, format='D', unit=self.angle_unit))

        if self.angle is not None:
            if self.windows == "SAMPLE":
                columns.append(fits.Column(
                    name='ANG', array=self.angle, format='D', unit=self.angle_unit))
            if self.windows == "CLBP":
                columns.append(fits.Column(
                    name='ANG', array=self.angle, format='2K', unit=self.angle_unit))
        if self.npairs is not None:
            columns.append(fits.Column(
                name='NPAIRS', array=self.npairs, format='D'))
        if self.varxi is not None:
            columns.append(fits.Column(
                name='VARXI', array=self.varxi, format='D'))
        if self.extra_cols is not None:
            for (colname, arr) in self.extra_cols.iteritems():
                columns.append(fits.Column(
                    name='XTRA_'+colname, array=arr, format='D'))

        extension = fits.BinTableHDU.from_columns(columns, header=header)
        return extension


class CovarianceMatrixInfo(object):
    """Encapsulate a covariance matrix and indices in it"""

    def __init__(self, name, names, lengths, covmat):
        super(CovarianceMatrixInfo, self).__init__()
        self.name = name
        self.names = names
        self.lengths = lengths
        self.starts = [0]
        for i, l in enumerate(self.lengths[:-1]):
            self.starts.append(l + self.starts[i])
        self.covmat = covmat
        self.diagonal = covmat.diagonal()

    def get_error(self, name):
        i = self.names.index(name)
        start = self.starts[i]
        end = start + self.lengths[i]
        return self.diagonal[start:end]**0.5

    def to_fits(self):
        header = fits.Header()
        
        header[COV_SENTINEL] = True
        header['EXTNAME'] = self.name
        for i, (start_index, name) in enumerate(zip(self.starts, self.names)):
            header['STRT_{}'.format(i)] = start_index
            header['NAME_{}'.format(i)] = name
        extension = fits.ImageHDU(data=self.covmat, header=header)
        return extension

    @classmethod
    def from_fits(cls, extension):
        cov_name = extension.name
        covmat = extension.data
        header = extension.header
        i = 0
        measurement_names = []
        start_indices = []
        while True:
            name_card = 'NAME_{}'.format(i)
            if name_card not in header:
                break
            measurement_names.append(header[name_card])
            start_index_card = 'STRT_{}'.format(i)
            start_indices.append(header[start_index_card])
            i += 1
        lengths = []
        current_length = 0
        # this only works if more than one spectrum
        if len(start_indices) > 1:
            for start, end in zip(start_indices[:-1], start_indices[1:]):
                lengths.append(end - start)
            if start_indices:
                lengths.append(covmat.shape[0] - start_indices[-1])
        else:
            lengths.append(covmat.shape[0])
        return cls(cov_name, measurement_names, lengths, covmat)

    @classmethod
    def from_spec_lists(cls, spec_lists, cov_name, mode='full'):
        """Often the covariance will be computed by measuring the statistic(s) in question
        on many simulated realisations of the dataset. This function takes a list of such 
        measurements, *each one a list SpectrumMeasurement objects*, computes the mean and covariance, and
        returns the mean as a list of SpectrumMeasurements, and the covariance as a CovarianceMatrixInfo
        object. mode should be one of full, subsample or jackknife"""
        # first check that spec_lists is a list of lists, and that there are at least 2
        print('spec_lists', spec_lists)
        try:
            spec_lists[1][0]
        except Exception as e:
            print("spec_lists should be a list of lists with at least two elements")
            raise(e)

        # Get measurement names and lengths from first element of spec_lists
        num_spec = len(spec_lists[0])
        names = [s.name for s in spec_lists[0]]
        lengths = [len(s.value) for s in spec_lists[0]]

        # Now loop through all realisations, building up list of numpy arrays of raw measurements
        n_real = len(spec_lists)
        spec_arrays = []
        for i_real in range(n_real):
            spec_array = []
            # check this realisation has right number,type,length of spectra
            for i_spec in range(num_spec):
                assert spec_lists[i_real][i_spec].name == names[i_spec]
                assert len(spec_lists[i_real][i_spec].value) == lengths[i_spec]
                spec_array += list(spec_lists[i_real][i_spec].value)
            spec_arrays.append(np.array(spec_array))

        # Now compute covariance
        spec_arrays = np.array(spec_arrays)
        cov_values, _ = sample_cov(spec_arrays, mode=mode)
        mean_spec_values = np.mean(spec_arrays, axis=0)

        # make list of mean 2pt specs by copying spec_lists[0] and replacing value column
        mean_spec = copy.copy(spec_lists[0])
        index_start = 0
        for i_spec in range(num_spec):
            end = index_start + lengths[i_spec]
            inds = np.arange(index_start, index_start + lengths[i_spec])
            index_start = end
            mean_spec[i_spec].value = mean_spec_values[inds]

        return cls(cov_name, names, lengths, cov_values), mean_spec


class TwoPointFile(object):
    def __init__(self, spectra, kernels, windows, covmat_info):
        if windows is None:
            windows = {}
        self.spectra = spectra
        dv_start = 0
        for s in self.spectra:
            n_dv = len(s.value)
            s.dv_index = np.arange(dv_start, dv_start+n_dv)
            dv_start += n_dv
        self.kernels = kernels
        self.windows = windows
        self.covmat_info = covmat_info
        if covmat_info:
            #self.covmat = covmat_info.covmat
            self.covmat = self.get_cov_start()
            for s in self.spectra:
                s.error = self.covmat_info.get_error(s.name)
        else:
            self.covmat = None
        self._spectrum_index = None

    def get_spectrum(self, name):
        spectra = [spectrum for spectrum in self.spectra if spectrum.name == name]
        n = len(spectra)
        if n == 0:
            raise ValueError("Spectrum with name %s not found in file" % name)
        elif n > 1:
            raise ValueError(
                "Multiple spectra with name %s found in file" % name)
        else:
            return spectra[0]

    def get_kernel(self, name):
        kernels = [kernel for kernel in self.kernels if kernel.name == name]
        n = len(kernels)
        if n == 0:
            raise ValueError("Kernel with name %s not found in file" % name)
        elif n > 1:
            raise ValueError(
                "Multiple kernel with name %s found in file" % name)
        else:
            return kernels[0]

    def _build_spectrum_index(self):
        index = 0
        self._spectrum_index = {}
        for spectrum in self.spectra:
            name = spectrum.name
            bin1 = spectrum.bin1
            bin2 = spectrum.bin2
            angbin = spectrum.angular_bin
            n = len(bin1)
            for i in range(n):
                self._spectrum_index[(
                    name, bin1[i], bin2[i], angbin[i])] = index
                index += 1

    def get_overall_index(self, spectrum_name, bin1, bin2, angbin):
        if self._spectrum_index is None:
            self._build_spectrum_index()
        return self._spectrum_index[(spectrum_name, bin1, bin2, angbin)]

    def _mask_covmat(self, masks):
        # Also cut down the covariance matrix accordingly
        if self.covmat is not None:
            mask = np.concatenate(masks)
            self.covmat = self.covmat[mask, :][:, mask]
            self.covmat_info = CovarianceMatrixInfo(self.covmat_info.name, [
                                                    s.name for s in self.spectra], [len(s) for s in self.spectra], self.covmat)

    def mask_bad(self, bad_value):
        "Go through all the spectra masking out data where they are equal to bad_value"
        masks = []
        # go through the spectra and covmat, masking out the bad values.
        for spectrum in self.spectra:
            # nb this will not work for NaN!
            mask = (spectrum.value != bad_value)
            spectrum.apply_mask(mask)
            print("Masking {} values in {}".format(
                mask.size - mask.sum(), spectrum.name))
            # record the mask vector as we will need it to mask the covmat
            masks.append(mask)
        if masks:
            self._mask_covmat(masks)

    def reorder_canonical(self):
        masks = []
        print("Reordering all data")
        n = 0
        for spectrum in self.spectra:
            mask = spectrum.canonical_order()
            spectrum.apply_mask(mask)
            masks.append(mask+n)
            n += len(mask)
        if masks:
            self._mask_covmat(masks)

    def mask_indices(self, spectrum_name, indices):
        s = self.get_spectrum(spectrum_name)
        mask_points = np.array(indices)
        masks = []
        for s in self.spectra:
            mask = np.ones(len(s), dtype=bool)
            if s.name == spectrum_name:
                mask[mask_points] = False
                s.apply_mask(mask)
            print("Keeping {} points in {}".format(mask.sum(), s.name))
            masks.append(mask)
            print(mask_points)
            print(mask)
        self._mask_covmat(masks)

    def mask_cross(self):
        masks = []
        for spectrum in self.spectra:
            mask = spectrum.auto_bins()
            spectrum.apply_mask(mask)
            print("Masking {} cross-values in {}".format(mask.size -
                                                         mask.sum(), spectrum.name))
            masks.append(mask)
        if masks:
            self._mask_covmat(masks)

    def mask_scales(self, cuts={}, bin_cuts=[]):
        masks = []
        for spectrum in self.spectra:
            mask = np.ones(len(spectrum), dtype=bool)
            for b1, b2 in spectrum.bin_pairs:
                w_full = np.where((spectrum.bin1 == b1) &
                                  (spectrum.bin2 == b2))[0]
                if (spectrum.name, b1, b2) in bin_cuts:
                    print("Removing {} bin ({},{}) altogether.".format(
                        spectrum.name, b1, b2))
                    mask[w_full] = False
                    continue

                cut = cuts.get((spectrum.name, b1, b2))
                if cut is None:
                    print("No cut specified for {} bin ({},{})".format(
                        spectrum.name, b1, b2))
                    continue

                # Actually do the cut
                ang_min, ang_max = cut
                w = np.where((spectrum.bin1 == b1) & (spectrum.bin2 == b2) &
                             ((spectrum.angle < ang_min) | (spectrum.angle > ang_max)))[0]

                print("Cutting {} bin pair ({},{}) to angle range ({} - {}):"
                      " this removes {} values out of {}".format(
                    spectrum.name, b1, b2, ang_min, ang_max, len(w), len(w_full)))

                mask[w] = False
            masks.append(mask)
            spectrum.apply_mask(mask)
            print("")

        if masks:
            self._mask_covmat(masks)

    def mask_scale(self, spectra_to_cut, min_scale=-np.inf, max_scale=np.inf):
        masks = []
        # go through the spectra and covmat, masking out the bad values.
        for spectrum in self.spectra:
            mask = np.ones(len(spectrum.value), dtype=bool)
            if (spectra_to_cut != "all") and (spectrum.name not in spectra_to_cut):
                masks.append(mask)
            else:
                # nb this will not work for NaN!
                mask = (spectrum.angle > min_scale) & (
                    spectrum.angle < max_scale)
                spectrum.apply_mask(mask)
                print("Masking {} values in {} because they had ell or theta outside ({},{})".format(
                    mask.size - mask.sum(), spectrum.name, min_scale, max_scale))
                # record the mask vector as we will need it to mask the covmat
                masks.append(mask)

        if masks:
            self._mask_covmat(masks)

    def choose_data_sets(self, data_sets):
        """Strip out any data sets not in the given list."""
        data_sets = [d.lower() for d in data_sets]
        mask = []
        use = []
        for spectrum in self.spectra:
            if spectrum.name.lower() in data_sets:
                use.append(True)
                mask.append(np.ones(spectrum.bin1.size, dtype=bool))
            else:
                use.append(False)
                mask.append(np.zeros(spectrum.bin1.size, dtype=bool))
        for data_set in data_sets:
            if not any(spectrum.name.lower() == data_set for spectrum in self.spectra):
                raise ValueError(
                    "Data set called {} not found in two-point data file.".format(data_set))
        self.spectra = [s for (u, s) in zip(use, self.spectra) if u]
        if self.covmat is not None:
            mask = np.concatenate(mask)
            self.covmat = self.covmat[mask, :][:, mask]

    def get_cov_start(self):

        # This gets the covariance array in the right order (before any scale cuts etc.)
        cov = self.covmat_info.covmat
        if self.covmat_info.names == [spec.name for spec in self.spectra]:
            # Ordering is already ok
            return cov
        print("Covariance matrix is not in the same order as the 2pt measurement extensions...doing some damn fiddly")
        print("re-ordering, if I screw it up it's your fault for not putting your covariance in the right order")
        cov_starts = self.covmat_info.starts
        cov_lengths = self.covmat_info.lengths
        cov_names = self.covmat_info.names
        cov_ends = [cov_lengths[0]]
        for i in range(len(cov_names) - 1):
            cov_ends.append(cov_ends[i] + cov_lengths[i + 1])

        assert cov_ends[-1] == cov.shape[0]

        total_l = 0
        spec_inds = []
        spec_names = [spec.name for spec in self.spectra]
        for spectrum in self.spectra:
            spec_inds.append(cov_names.index(spectrum.name))
            total_l += cov_lengths[cov_names.index(spectrum.name)]
        cov_out = np.zeros((total_l, total_l))
        start_i = 0

        for ti, ind_i in zip(spec_names, spec_inds):
            start_j = 0
            for tj, ind_j in zip(spec_names, spec_inds):
                cov_out[start_i:start_i + cov_lengths[ind_i], start_j:start_j + cov_lengths[ind_j]
                        ] = cov[cov_starts[ind_i]:cov_ends[ind_i], cov_starts[ind_j]:cov_ends[ind_j]]
                start_j += cov_lengths[ind_j]
            start_i += cov_lengths[ind_i]
        return cov_out

    def to_fits(self, filename, overwrite=False, clobber=False):
        hdus = [fits.PrimaryHDU()]

        if clobber:
            warnings.warn("The 'clobber' keyword in twopoint is deprecated."
                          "Please switch to overwrite.", DeprecationWarning)

        if self.covmat_info is not None:
            hdus.append(self.covmat_info.to_fits())

        for spectrum in self.spectra:
            if spectrum.windows not in window_types:
                raise NotImplementedError(
                    "Sorry - not yet coded general case with ell/theta window functions")
            hdus.append(spectrum.to_fits())

        if self.kernels is not None:
            for kernel in self.kernels:
                hdus.append(kernel.to_fits())

        hdulist = fits.HDUList(hdus)
        hdulist.writeto(filename, overwrite=(clobber or overwrite))

    @classmethod
    def from_fits(cls, filename, covmat_name="COVMAT"):
        fitsfile = fits.open(filename)
        spectra = []
        kernels = []
        windows = {}

        # Load the covariance matrix from the file, typically stored as
        # image data.
        # It turns out it is more conventient to load this first.
        # We can also use no covmat at all.
        if covmat_name is None:
            covmat_info = None
            covmat = None
        else:
            extension = fitsfile[covmat_name]
            covmat_info = CovarianceMatrixInfo.from_fits(extension)

        # First load all the spectra in the file
        # Spectra are indicated by the "2PTDATA" keyword
        # in the header being "T"
        for extension in fitsfile:
            if extension.header.get(TWOPOINT_SENTINEL):
                spectra.append(SpectrumMeasurement.from_fits(
                    extension, covmat_info))

        # Each spectrum needs kernels, usually some n(z).
        # These were read from headers when we loaded the 2pt data above above.
        # Now we are loading those kernels into a dictionary
        for extension in fitsfile:
            if extension.header.get(NZ_SENTINEL):
                kernels.append(NumberDensity.from_fits(extension))

        # We might also require window functions W(ell) or W(theta). It's also possible
        # that we just use a single sample value of ell or theta instead.
        # If the spectra required it (according to the header) then we also
        # need to load those in.
        for spectrum in spectra:
            if spectrum.windows not in window_types and spectrum.windows not in windows:
                windows[spectrum.windows] = cls._windows_from_fits(
                    fitsfile[windows])

        # return a new TwoPointFile object with the data we have just loaded
        return cls(spectra, kernels, windows, covmat_info)

    @classmethod
    def _windows_from_fits(cls, extension):
        raise NotImplementedError("non-sample window functions in ell/theta")

    def plots(self, root, colormap='viridis', savepdf=False, latex=True, plot_spectrum=True, plot_kernel=True, plot_cov=True, cov_vmin=None, save_pickle=False, load_pickle=False, remove_pickle=True, label_legend='', blind_yaxis=False, callback=None):
        """
        Makes plot of each for your spectra, kernels and covariance. Allows you to compare the spectra of different files. 
        Options:
        - root: Name of the output plots.
        - colormap: Colormap used for the plots.
        - savepdf: True if you want to save pdf too, besides png.
        - latex: True if want to save with latex font. It will be slower. Set to false to test plot.
        - plot_spectrum, plot_kernel, plot_cov: whether or not to make this plots
        - cov_vmin = minimum value for the colorbar in the covariance plot.
        - plot_spectrum, plot_kernel and plot_cov are boolean variables that are true if you want to make these plots.
        - save_pickle: if true it saves a pickle file to edit be able to compare different files.
        - load_pickle: if true, it will continue the plot starting from a pickle file.
        - remove_pickle: set to true if you want to keep this file to edit your plot afterwards.
        - label_legend: name that will appear in the legend when comparing different files.
        - blind_axis: True if you want to remove the y-axis labels. 
        """

        import matplotlib.pyplot as plt
        from matplotlib import ticker
        import pickle as pl

        if latex:
            plt.rc('text', usetex=True)
            plt.rc('font', family='serif')

        def savefig(name):
            print("Saving {}".format(name))
            plt.savefig(name, bbox_inches='tight', dpi=400)
            if savepdf:
                plt.savefig(name + '.pdf', bbox_inches='tight')

        def corr_names(spectrum):
            '''
            Get latex labels and colors for each kind of correlation.
            '''
            if (spectrum.type1 == spectrum.type2 == Types.galaxy_shear_plus_real):
                corr_type = 'xip'
                label = r"$\xi_+(\theta)$"
                color = plt.get_cmap(colormap)(0)

            elif (spectrum.type1 == spectrum.type2 == Types.galaxy_shear_minus_real):
                corr_type = 'xim'
                label = r"$\xi_-(\theta)$"
                color = plt.get_cmap(colormap)(0.2)

            elif (spectrum.type1 == Types.galaxy_position_real) & (spectrum.type2 == Types.galaxy_shear_plus_real):
                corr_type = 'gt'
                label = r"$\gamma_t(\theta)$"
                color = plt.get_cmap(colormap)(0.4)

            elif (spectrum.type1 == spectrum.type2 == Types.galaxy_position_real):
                corr_type = 'wtheta'
                label = r"$w(\theta)$"
                color = plt.get_cmap(colormap)(0.6)

            else:
                corr_type = None
                label = None
                color = None

            return corr_type, label, color

        if plot_spectrum:

            for spectrum in self.spectra:

                corr_type, label, color = corr_names(spectrum)

                if corr_type is None:
                    continue

                mtype = 'o'
                # Use one color for each file in case you want to compare different files
                if save_pickle:
                    color = plt.get_cmap(colormap)(0)
                if save_pickle and load_pickle:
                    color = plt.get_cmap(colormap)(0.2)
                if load_pickle:
                    color = plt.get_cmap(colormap)(0.4)

                name = "{}_{}.png".format(root, spectrum.name)
                pairs = spectrum.bin_pairs
                npairs = len(pairs)
                bins1 = np.transpose(pairs)[0]
                bins2 = np.transpose(pairs)[1]
                nbins1 = np.max(bins1)
                nbins2 = np.max(bins2)
                if npairs == 0:
                    continue

                # Choose different figure sizes depending on the number of redshift bins
                if not all(bins1 == bins2):
                    fig, ax = plt.subplots(nbins2, nbins1, figsize=(
                        1.6*nbins1, 1.6*nbins2), sharey=True, sharex=True)

                if all(bins1 == bins2):
                    # Autocorrelation only will have a different figure size and structure
                    fig, ax = plt.subplots(1, nbins1, figsize=(
                        1.6*nbins1, 1.4), sharey=True, sharex=True)
                    ax = np.diag(ax)

                if load_pickle:
                    # If we are continuing a plot to compare different files, load the fig and axes objects
                    name_pickle = "{}_{}.pickle".format(root, spectrum.name)
                    fig = pl.load(open(name_pickle, 'rb'))
                    if remove_pickle:
                        os.system('rm %s' % name_pickle)
                    ax = fig.axes
                    ax = np.array(ax)
                    if len(ax) == nbins1:
                        ax = np.diag(ax)
                    else:
                        ax = np.reshape(ax, (nbins2, nbins1))

                for k, pair in enumerate(pairs):

                    i, j = pair
                    theta, xi = spectrum.get_pair(i, j)
                    error = spectrum.get_error(i, j)

                    ax[j-1][i-1].errorbar(theta, abs(xi), yerr=error, fmt=mtype, capsize=1.5,
                                          markersize=3, color=color, mec=color, elinewidth=1., label=label_legend)

                    ax[j-1][i-1].text(0.85, 0.85, "{},{}".format(i, j), horizontalalignment='center',
                                      verticalalignment='center', transform=ax[j-1][i-1].transAxes, fontsize=12)
                    ax[j-1][i-1].set_xscale('log', nonposx='clip')
                    ax[j-1][i-1].set_yscale('log', nonposy='clip')
                    ax[j-1][i-1].xaxis.set_major_formatter(
                        ticker.FormatStrFormatter('$%d$'))
                    if blind_yaxis:
                        ax[j-1][i-1].yaxis.set_ticklabels([])

                    if (not all(bins1 == bins2)) & (j == nbins2):
                        ax[j-1][i-1].set_xlabel(r"$\theta$ [arcmin]")
                    if all(bins1 == bins2):
                        ax[j-1][i-1].set_xlabel(r"$\theta$ [arcmin]")
                    if i == 1:
                        ax[j-1][i-1].set_ylabel(label)

                    if not save_pickle:
                        if (not all(bins1 == bins2)) & (corr_type != 'gt') & (j > i):
                            fig.delaxes(ax[i-1, j-1])

                if not save_pickle:
                    plt.legend(prop={'size': 5})
                    savefig(name)
                if save_pickle:
                    name_pickle = "{}_{}.pickle".format(root, spectrum.name)
                    pl.dump(fig, file(name_pickle, 'w'))

                plt.close()
            plt.close('all')

        if plot_kernel:
            for kernel in self.kernels:
                name = "{}_{}.png".format(root, kernel.name)
                plt.figure()
                fig, ax = plt.subplots(1, 1, figsize=(5, 3))
                for i, nz in enumerate(kernel.nzs):
                    color = color = plt.get_cmap(colormap)(0.2*i)
                    ax.plot(kernel.z, nz, lw=1.5, color=color)
                    ax.fill_between(kernel.z, 0, nz, color=color, alpha=0.2)
                ax.set_xlabel('Redshift')
                ax.set_ylabel('Normalized counts')
                ax.set_xlim(0, 2)
                ax.set_ylim(bottom=0)
                plt.tight_layout()
                savefig(name)

        if plot_cov:
            def corrmatrix(cov):
                cov = np.mat(cov)
                D = np.diag(np.sqrt(np.diag(cov)))
                d = np.linalg.inv(D)
                corr = d*cov*d
                return corr

            name = "{}_{}.png".format(root, 'cov')
            cov = self.covmat
            ncov1 = len(cov)
            ncov2 = len(cov[0])

            corr = corrmatrix(cov)
            if cov_vmin is None:
                cov_vmin = np.min(corr)

            figsize1 = 1.22222222227*ncov1/100.
            figsize2 = ncov2/100.
            fig, ax = plt.subplots(1, 1, figsize=(figsize1, figsize2))
            im = ax.imshow(corr, interpolation='nearest',
                           aspect='auto', origin='lower', vmin=cov_vmin, vmax=1., cmap=colormap+'_r')
            cbar = fig.colorbar(im)

            # Get labels to put in the title
            labels = ''
            for spectrum in self.spectra:
                corr_type, label, color = corr_names(spectrum)
                if corr_type is not None:
                    labels = labels + label + ' $|$ '
            labels = labels[:-4]
            ax.set_title(labels)

            # Plot lines to divide covariance
            lengths = self.covmat_info.lengths
            pos_lines = [0]
            for i in range(len(lengths)):
                pos_lines.append(pos_lines[i] + lengths[i])
            pos_lines = pos_lines[1:-1]
            for line in pos_lines:
                ax.axvline(x=line, c='k', lw=1, ls='-')
                ax.axhline(y=line, c='k', lw=1, ls='-')

            savefig(name)


class SpectrumCovarianceBuilder(object):
    """
    This class helps you ensure that the ordering between a set of data points and 
    their covariance is consistently maintained.  You add data points to it one by one.,
    in the order that they appear in the covariance.

    Here is an example using ther CFHTLenS revisited files:

    >>> theta_values = [1.41, 2.79, 5.53, 11.0, 21.7, 43.0, 85.2]
    >>> types = {
        'xip': twopoint.Types.galaxy_shear_plus_real,
        'xim': twopoint.Types.galaxy_shear_minus_real,
    }
    >>> kernel='NZ_SOURCE'
    >>> builder = twopoint.SpectrumCovarianceBuilder()
    >>> i_bin, data, jk_err, sim_err = np.loadtxt("./xipm_cfhtlens_regcomb_blind1_passfields_athenazsj.dat").T
    >>> i = 0
    >>> for bin1 in range(7):
        for bin2 in range(bin1,7):
            for name in ['xip', 'xim']:
                stype = types[name]
                for angbin in range(1,8):
                    ang = theta_values[angbin-1]
                    value = data[i]
                    i+=1
                    builder.add_data_point(kernel,kernel,stype,stype,bin1+1,bin2+1,ang,angbin,value)
    >>> names = {builder.types[0]:"xip", builder.types[1]:"xim"}
    >>> builder.set_names(names)
    >>> spectra, covmat_info = builder.generate(covmat,"arcmin")
    """

    def __init__(self):
        self.kernel1 = []
        self.kernel2 = []
        self.bin1 = []
        self.bin2 = []
        self.type1 = []
        self.type2 = []
        self.ang = []
        self.angbin = []
        self.value = []
        self.names = None
        self.types = []
        self.total_length = 0

    def add_data_point(self, kernel1, kernel2, type1, type2, bin1, bin2, ang, angbin, value):
        self.kernel1.append(kernel1)
        self.kernel2.append(kernel2)
        self.type1.append(type1)
        self.type2.append(type2)
        self.bin1.append(bin1)
        self.bin2.append(bin2)
        self.ang.append(ang)
        self.angbin.append(angbin)
        self.value.append(value)
        self.total_length += 1

        spec = kernel1, kernel2, type1, type2
        if spec not in self.types:
            self.types.append(spec)

    def _freeze(self):
        self.bin1 = np.array(self.bin1)
        self.bin2 = np.array(self.bin2)
        self.ang = np.array(self.ang)
        self.angbin = np.array(self.angbin)
        self.value = np.array(self.value)

    def set_names(self, names):
        for t in self.types:
            if t not in names:
                raise ValueError(
                    "Please provide name for spectrum of type: {}".format(t))
        self.names = names

    def generate(self, covariance, angle_unit):
        if self.names is None:
            raise ValueError(
                "Please provide names for each type in self.types first")
        self._freeze()
        assert covariance.shape == (self.total_length, self.total_length)
        # get all the unique pairs of type1,type2.  maintain order
        # shouldn't be more than a few types so the list membership test is fast
        master_index_vector = []
        spectra = []
        for kernel1, kernel2, type1, type2 in self.types:
            kernels = (kernel1, kernel2)
            types = (type1, type2)
            spectrum_index_vector = []
            for i in range(self.total_length):
                k1 = self.kernel1[i]
                k2 = self.kernel2[i]
                t1 = self.type1[i]
                t2 = self.type2[i]
                if (type1 != t1) or (type2 != t2) or (kernel1 != k1) or (kernel2 != k2):
                    continue
                master_index_vector.append(i)
                spectrum_index_vector.append(i)
            bins = (self.bin1[spectrum_index_vector],
                    self.bin2[spectrum_index_vector])
            angular_bin = self.angbin[spectrum_index_vector]
            value = self.value[spectrum_index_vector]
            angle = self.ang[spectrum_index_vector]
            name = self.names[(kernel1, kernel2, type1, type2)]
            windows = "SAMPLE"
            spectrum = SpectrumMeasurement(
                name, bins, types, kernels, windows, angular_bin, value,
                angle=angle, angle_unit=angle_unit)
            spectra.append(spectrum)

        reordered_covariance = covariance[master_index_vector][:, master_index_vector]
        covmat_info = CovarianceMatrixInfo("COVMAT", [s.name for s in spectra], 
                                           [len(s) for s in spectra], reordered_covariance)

        return spectra, covmat_info


if __name__ == '__main__':
    import sys
    filename = sys.argv[1]
    output_root = sys.argv[2]
    T = TwoPointFile.from_fits(filename, covmat_name=None)
    T.plots(output_root)
