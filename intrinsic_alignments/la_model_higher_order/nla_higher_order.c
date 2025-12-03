// Code to calculate higher-order terms for the (non)linear alignment model
// Based on eq 16 of Hirata & Seljak (2004) arXiv:astro-ph/0406275 
// and the correction in Hirata and Seljak (2010)
// Simon Samuroff 01/16 

#include "datablock/c_datablock.h"
#include "gsl/gsl_spline.h"
#include "utils.h"
#include <stdio.h>
#include "limber.h"
#include <math.h>

#define NEWTON_G 6.67408e-11
#define MSUN 1.98855e30
#define PI 3.14159
#define C1 5.0e-14

//Interpolator functions 
Interpolator2D * init_interp_2d_akima_grid(double *x1, double *x2, double **y, int N1, int N2);
void destroy_interp_2d(Interpolator2D * interp2d);
double interp_2d(double x1, double x2, Interpolator2D * interp2d);

typedef struct options_config{
	bool use_nla_model; 
} options_config;

void * setup(c_datablock * options){
	DATABLOCK_STATUS status;
	options_config * config = malloc(sizeof(options_config));

	status |= c_datablock_get_bool_default(options, OPTION_SECTION, "nla_model", true,
		&(config->use_nla_model));
	return config;
}

double f_E(double kx, double ky, double k){
	return (kx*kx - ky*ky)/ (k*k);
}

double f_B(double kx, double ky, double k){
	return 2.*kx*ky/ (k*k);
}

int execute(c_datablock * block, void * config_in)
{
	DATABLOCK_STATUS status=0;

	options_config * config = (options_config*) config_in;

	double * D, **b_g, **p_k, **P_II, **P_B;
	double * zbg, *kbg;
	double *z, *k;
	int nz1, nk1, nz, nk;
	int nbin;

	double omega_m, h;

	// Load the arrays and interpolators needed
	gsl_spline * DZ = load_spline(block, "growth_parameters", "z", "d_z");

	// Galaxy bias (assumed to be scale independent)
	status |= c_datablock_get_double_grid(block, "bias_field", "k_h", &nk, &kbg,  "z", &nz1, &zbg,  "b_g", &b_g);
	gsl_spline * BG = spline_from_arrays(nz1, zbg, b_g[0]);

	// Matter power spectrum
	char spectrum_section[60];
	if (config->use_nla_model) { 
		snprintf(spectrum_section, 60, "matter_power_nl" );}
	else{ 
		snprintf(spectrum_section, 60, "matter_power_lin" );}
	Interpolator2D * PK = load_interpolator(block, spectrum_section, "k_h", "z", "P_k");
	status |=  c_datablock_get_double_array_1d(block, spectrum_section, "z", &z, &nz);
	status |=  c_datablock_get_double_array_1d(block, spectrum_section, "k_h", &k, &nk);

	status |= c_datablock_get_double_grid(block, spectrum_section, "k_h", &nk, &k, "z", &nz, &z, "p_k", &p_k);
	P_II= (double **)malloc(nk*sizeof(double*));
		P_II[0] = (double *)malloc(nk*nz*sizeof(double));
		for(int i= 1 ; i< nk ; i++){
			P_II[i]= P_II[0]+i*nz;}

	P_B= (double **)malloc(nk*sizeof(double*));
		P_B[0] = (double *)malloc(nk*nz*sizeof(double));
		for(int i= 1 ; i< nk ; i++){
			P_B[i]= P_B[0]+i*nz;}

	// Get cosmological parameters and calculate normalisation
	status |=  c_datablock_get_double(block, "cosmological_parameters", "Omega_m", &omega_m);
	status |=  c_datablock_get_double(block, "cosmological_parameters", "h0", &h);

	double rho_crit0 = 3000. * h * h / (8. * PI * NEWTON_G);
	double A = C1 * C1 * MSUN * MSUN * omega_m * omega_m * rho_crit0 * rho_crit0;
	printf("Calculated IA normalisation: %e (status: %d)", A, status);

	// Define the sampling for the integral, uniformly spaced in log space
	int nkf = 1000;
	double kmin = 1.0e-3;
	double kmax = 1.0e3;
	double krange[nkf];
	double dlogk = (log(kmax)-log(kmin))/nkf;
	for (int i=0 ; i<nkf ; ++i){
		krange[i] = exp(log(kmin) + i*dlogk);
	}

	// Then do the integral itself

	// i1, i2, i3 count the values of the dummy variables k1x, k1y, k1z
	// j counts the redshift points
	// k0  is the fixed wavenumber for which the integral is being evaluated
	double k0, k1, k2; 
	double k1x, k1y, k1z;
	double k2x, k2y, k2z;
	int i1, i2, i3, j;

	double coeff = 0. ;//bg * bg * C1 * C1 * rhocrit * Omega_m * Omega_m / Dz / Dz / (2.*PI)**3;
	double b=0, d=0;
	double tmp_E, integrand_E;
	double tmp_B, integrand_B;
	double pk1, pk2, f1, f2;
	double fB1, fB2;
	// Loop over redshift
	for (int j=0 ; j<nz ; ++j){
		b = gsl_spline_eval(BG, z[j], NULL);
		d = gsl_spline_eval(DZ, z[j], NULL);
		coeff = b * b * A /(d * d);
		
		// Then wavenumber
		for (int i=0 ; i<nk ; ++i){
			k0 = k[i];
			// For each z,k combination evaluate a 3D volume integral
			tmp_E = 0.0;
			integrand_E = 0.0 ;
			tmp_B = 0.0;
			integrand_B = 0.0 ;
			for (int iz = 0 ; iz<nkf ; ++iz){
				k1z = krange[iz];
				for (int iy = 0 ; iy<nkf ; ++iy){
					k1y = krange[iy];
					for (int ix=0 ; ix<nkf ; ++ix){

						k1x = krange[ix];

						// Since k is defined to be parallel to the x axis as in Hirata and 
						// Seljak (2004) all components of the k2 vector can be computed from k1x
						k2x = k0 - k1x;
						k2y = -1. * k1y;
						k2z = -1. * k1z;

						k1 = sqrt(k1x*k1x + k1y*k1y + k1z*k1z);
						k2 = sqrt(k2x*k2x + k2y*k2y + k2z*k2z);
		
						pk1 = interp_2d(k1, z[j], PK);
						pk2 = interp_2d(k2, z[j], PK);

						f1 = f_E(k1x, k1y, k1);
						f2 = f_E(k2x, k2y, k2);

						tmp_E = f2*(f1+f2)*pk1*pk2;
						// An extra factor of k is needed since we're integrating wrt logk
						// Include one for each dimension of the volume integral
						tmp_E = tmp_E * k1x * k1y * k1z * dlogk * dlogk *dlogk;
						integrand_E += tmp_E;

						// Also do the B mode spectrum as this is of the same order
						// This is very similar to the higher-order E mode term but with a
						// slightly different dependence on k1 and k2
						fB1 = f_B(k1x, k1y, k1);
						fB2 = f_B(k2x, k2y, k2);
						tmp_B = fB2*(fB1+fB2)*pk1*pk2;
						tmp_B = tmp_B * k1x * k1y * k1z * dlogk * dlogk *dlogk;
						integrand_B += tmp_B;
		
					}
				}
			}
			P_II[j][i] = coeff * integrand_E;
			P_B[j][i] = coeff * integrand_B;
			printf ("k=%e z=%e P_II[%d][%d]=%e",k0,z[j], j, i, P_II[j][i]); 

		}
	}

	//Finally save to the datablock
	status|=c_datablock_put_double_grid(block, IA_SPECTRUM_II_SECTION, "k_h", nk, k_h, "z", nz, z, "P_II_EEho", P_II);
	status|=c_datablock_put_double_grid(block, IA_SPECTRUM_II_SECTION, "k_h", nk, k_h, "z", nz, z, "P_II_BB", P_B);


	destroy_interp_2d(PK);
	gsl_spline_free(BG);
	gsl_spline_free(DZ);
	free(D);
	free(b_g);
	free(p_k);
	free(P_II);
	free(zbg);
	free(kbg);
	free(z);
	free(k);
	
  	return status;

}

int cleanup(void * config)
{
	// Free the memory that we allocated in the
	// setup
	free(config);
}
