"""
Author: Suano Sugiyama

Product calibration factor for delta_sigma, wp, and rp because of 
mean redshift shifts and change in Omm and w0. 

Code reviewed by Tianqing Zhang in Oct 2023
"""


from cosmosis.datablock import names, option_section
from astropy.io import fits
from meascorr import dSigma_meascorr_class, wp_meascorr_class, rp_meascorr_class


class dSigma_meascorr_interface:
    def __init__(self, options):
        # See ${COSMOSIS_DIR}/number_density/photoz_bias/photoz_bias.py 
        # for the consistency of this section refenrece
        sample = options.get_string(option_section, "sample", "")
        bias_section = options.get_string(option_section, "bias_section", "")
        per_bin = options.get_bool(option_section, "per_bin", True)
        if bias_section == "" and sample == "":
            bias_section = "wl_photoz_errors"
        elif bias_section == "":
            bias_section = sample + "_errors"
        self.bias_section = bias_section
        self.per_bin = per_bin
        self.section_name = options.get_string(option_section, "output_section", "f_ds")
        
        Omm = options.get_double(option_section, "Omm", None)
        w0 = options.get_double(option_section, "w0", None)
            
        # Instantiate the dSigma measurement correction handler
        fname_fits = options.get_string(option_section, 'sumwlssigcritinvPz_file', "")
        self.load_fits(fname_fits)
       
        
        if (Omm != None) and (w0 != None):
            for mc in self.dSigma_corrs:
                mc.config['Omm'] = Omm
                mc.config['w0'] = w0
        
    def load_fits(self, fname_fits):
        hdus = fits.open(fname_fits)
        
        # get the photo-z bin
        self.zs = hdus['PZBIN'].data['Z_MID']
        
        # get the matrix of sumwlssigcritinvPz and zl 
        # and make dSigma_meascorr instance
        self.dSigma_corrs = []
        for ext in hdus:
            if not ext.header.get('SUMPZ', False):
                continue
            mc = dSigma_meascorr_class.from_fits(ext)
            mc.set_zs(self.zs)
            self.dSigma_corrs.append(mc)
        
    def execute(self, block):
        biases = self.bias_section
        Omm = block[names.cosmological_parameters, 'omega_m']
        w0  = block[names.cosmological_parameters, 'w']
        for mc in self.dSigma_corrs:
            # Get bin0, bin1 for this correction module
            # where bin0 is the lens bin id, 
            # and bin1 is the source bin id.
            bin0, bin1 = mc.get_bin_pair()
            dz = block[biases, 'bias_{0}'.format(bin1+1)]
            z, nz = block['nz_source', 'z' ], block['nz_source', 'bin_%d'%(bin1+1)]
            f = mc.get_corr_factor(dz, Omm, w0, bin1, z, nz)
            # print(bin0+1, bin1+1, f)
            block[self.section_name, 'bin_{0}_{1}'.format(bin0+1, bin1+1)] = f
    
    @classmethod
    def to_fits(cls, fname_fits, overwrite=False, clobber=False):
        hdus = [fits.PrimaryHDU()]
        
        if clobber:
            warnings.warn("The 'clobber' keyword in twopoint is deprecated."
                          "Please switch to overwrite.", DeprecationWarning)
            
        # photo-z bin
        header = fits.Header()
        header['EXTNAME'] = 'PZBIN'
        columns = [
            fits.Column(name='Z_MID', array=self.zs, format='D'),
        ]
        hdus.append(fits.BinTableHDU.from_columns(columns, header=header))
        
        # sumwlssigcritinvPz
        for mc in self.dSigma_corrs:
            huds.append(mc.to_fits())
        
        hdulist = fits.HDUList(hdus)
        hdulist.writeto(fname_fits, overwrite=(clobber or overwrite))


class wp_meascorr_interface:
    def __init__(self, options):
        """
        
        """
        # representative redshift
        self.redshifts  = options.get_double_array_1d(option_section, 'redshifts')
        config = {'Omm': options[option_section, 'Omm'], 'w0': options[option_section, 'w0']}
        self.wp_meascorr = wp_meascorr_class(config)
        self.section_name = options.get_string(option_section, "output_section", "f_wp")
    
    def execute(self, block):
        Omm = block[names.cosmological_parameters, 'omega_m']
        w0  = block[names.cosmological_parameters, 'w']
        for binid, redshift in enumerate(self.redshifts):
            f = self.wp_meascorr.get_corr_factor(redshift, Omm, w0)
            block[self.section_name, 'bin_{0}'.format(binid+1)] = f
        
        
class rp_meascorr_interface:
    def __init__(self, options):
        self.redshifts  = options.get_double_array_1d(option_section, 'redshifts')
        config = {'Omm': options[option_section, 'Omm'], 'w0': options[option_section, 'w0']}
        self.rp_meascorr = rp_meascorr_class(config)
        self.section_name = options.get_string(option_section, "output_section", "f_rp")
    
    def execute(self, block):
        Omm = block[names.cosmological_parameters, 'omega_m']
        w0  = block[names.cosmological_parameters, 'w']
        for binid, redshift in enumerate(self.redshifts):
            f = self.rp_meascorr.get_corr_factor(redshift, Omm, w0)
            block[self.section_name, 'bin_{0}'.format(binid+1)] = f
    
        
def setup(options):
    # extract the type of correction
    # type must be one of 
    #  - ds
    #  - wp
    #  - rp
    corr_type = options.get_string(option_section, "type")
    
    if corr_type == 'ds':
        return dSigma_meascorr_interface(options)
    elif corr_type == 'wp':
        return wp_meascorr_interface(options)
    elif corr_type == 'rp':
        return rp_meascorr_interface(options)

def execute(block, config):
    config.execute(block)
    return 0