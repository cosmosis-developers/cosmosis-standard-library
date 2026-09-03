/*CosmoSIS interface file for going from shear C(l) to xi+/-
Uses function tpstat_via_hankel from Martin Kilbinger's nicaea
*/
#include "datablock/c_datablock.h"
#include "datablock/section_names.h"
#include "maths.h"

#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <string.h>

const char * shear_xi = SHEAR_XI_SECTION;
const char * shear_cl = SHEAR_CL_SECTION;
const char * wl_nz = WL_NUMBER_DENSITY_SECTION;

typedef enum {shear_shear=0, matter=1, ggl=2} corr_type_t;
typedef enum {corrfct=0, map=1} filter_type_t;

const char * corr_type_names[3] = {"2-2", "0-0", "0-2"};
const char * filter_type_names[2] = {"correlation", "aperture"};

typedef struct cl_to_xi_config {
	char * input_section;
	char * output_section;
	corr_type_t corr_type;
	filter_type_t filter_type;

} cl_to_xi_config;


void * setup(c_datablock * options)
{
	cl_to_xi_config * config = malloc(sizeof(cl_to_xi_config));
	int corr_type,filter_type;
	int status = 0;
	bool auto_corr;

	char *filter_string=malloc(10*sizeof(char));
	char *output_string=malloc(100*sizeof(char));

	status |= c_datablock_get_int_default(options, OPTION_SECTION, "corr_type", 0, &corr_type);

	status |= c_datablock_get_int_default(options, OPTION_SECTION, "filter_type", 0, &filter_type);

	if (filter_type==corrfct) {
	  sprintf(filter_string,"xi");
	}
	else if (filter_type==map) {
	  sprintf(filter_string,"map");
	}
	else {
	  fprintf(stderr, "Unknown filter type in cl_to_xi (%d).\n",filter_type);
	  status = 1;
	}

	if (corr_type==shear_shear){
	  status |= c_datablock_get_string_default(options, OPTION_SECTION, "input_section_name", "shear_cl", &(config->input_section));

	  sprintf(output_string,"shear_%s",filter_string);		
	  status |= c_datablock_get_string_default(options, OPTION_SECTION, "output_section_name", output_string, &(config->output_section));

	}
	else if (corr_type==ggl){
	  status |= c_datablock_get_string_default(options, OPTION_SECTION, "input_section_name", "galaxy_shear_cl", &(config->input_section));

	  sprintf(output_string,"galaxy_shear_%s",filter_string);		
	  status |= c_datablock_get_string_default(options, OPTION_SECTION, "output_section_name", output_string, &(config->output_section));

	}
	else if (corr_type==matter){
	  status |= c_datablock_get_string_default(options, OPTION_SECTION, "input_section_name", "galaxy_cl", &(config->input_section));

	  sprintf(output_string,"galaxy_%s",filter_string);		
	  status |= c_datablock_get_string_default(options, OPTION_SECTION, "output_section_name", output_string, &(config->output_section));

	}
	else{
	  fprintf(stderr, "Unknown corr_type in cl_to_xi (%d). It should be one of %d (shear-shear), %d (shear-galaxy) or %d (position-galaxy).\n",corr_type,shear_shear,ggl,matter);
	  status = 1;
	}

    printf("Will Hankel transform %s -> %s with filter type '%s' and spins %s\n",
      config->input_section,
      config->output_section,
      filter_type_names[filter_type],
      corr_type_names[corr_type]
      );
	//auto_corr tells us whether we have an auto-correlation or cross-correlation.
	status |= c_datablock_get_bool_default(options, OPTION_SECTION, "auto_corr", true, &auto_corr);

	config->corr_type = (corr_type_t)corr_type;
	config->filter_type = (filter_type_t)filter_type;

	if (status){
	  fprintf(stderr, "Please specify input_section_name, output_section_name, filter_type=0,1, and corr_type=0,1, or 2 in the cl_to_xi module.\n");
	  exit(status);
	}

	free(filter_string);
	free(output_string);
	return config;
}

typedef struct p_projected_data{
	interTable * cl_table;
} p_projected_data;

static double P_projected_loglog(void *info, double ell, int bin_i, int bin_j, error ** err)
{
	p_projected_data * data = (p_projected_data*) info;
	interTable * cl_table = data->cl_table;
	double cl;
	cl =  exp(interpol_wr(cl_table, log(ell), err));
	return cl;
}

static double P_projected_logl(void *info, double ell, int bin_i, int bin_j, error ** err)
{
	p_projected_data * data = (p_projected_data*) info;
	interTable * cl_table = data->cl_table;
	//FILE * f = data->f;
	double cl;
	// if (ell<20.0) cl = 0.0;
	// else if (ell>3000.0) cl = 0.0;
	// else
	cl =  interpol_wr(cl_table, log(ell), err);
	//fprintf(f,"%le  %le\n", ell, cl);
	return cl;
}

int copy_metadata(int status, c_datablock * block, const char* output_section,
                  const char* input_section){

    //save useful metadata to the block
    //all just copied over from the cl_section actually.
    // check in each case if read successfully before trying to copy
    //number of z bins

    // keep track of the individual statuses as well as the overall one
    int s;

    int nbin_a;
    s = c_datablock_get_int(block, input_section, "nbin_a", &nbin_a);
    if (s == 0) s |= c_datablock_put_int(block, output_section, "nbin_a", nbin_a);
    status |= s;

    int nbin_b;
    s = c_datablock_get_int(block, input_section, "nbin_b", &nbin_b);
    if (s == 0) s |= c_datablock_put_int(block, output_section, "nbin_b", nbin_b);
    status |= s;

    //is_auto
    bool is_auto;
    s = c_datablock_get_bool(block, input_section, "is_auto", &is_auto);
    if (s == 0) s |= c_datablock_put_bool(block, output_section, "is_auto", is_auto);
    status |= s;

    //auto_only
    bool auto_only;
    s = c_datablock_get_bool(block, input_section, "auto_only", &auto_only);
    if (s == 0) s |= c_datablock_put_bool(block, output_section, "auto_only", auto_only);
    status |= s;

    //sample names
    char * sample_a;
    s = c_datablock_get_string(block, input_section, "sample_a", &sample_a);
    if (s == 0) s |= c_datablock_put_string(block, output_section, "sample_a", sample_a);
    status |= s;

    char * sample_b;
    s = c_datablock_get_string(block, input_section, "sample_b", &sample_b);
    if (s == 0) s |= c_datablock_put_string(block, output_section, "sample_b", sample_b);
    status |= s;

    //save name
    char * save_name;
    s = c_datablock_get_string(block, input_section, "save_name", &save_name);
    if (s == 0) s |= c_datablock_put_string(block, output_section, "save_name", save_name);
    status |= s;

    return status;
}


int execute(c_datablock * block, void * config_in)
{
    DATABLOCK_STATUS status=0;
    int num_z_bin_A;
    int num_z_bin_B;
    int count=2;
    double ** xi = malloc(count*sizeof(double*));
    for (int i=0; i<count; i++) xi[i] = malloc(sizeof(double)*N_thetaH);

    const int slen = 128;
    char output_section[slen];

    cl_to_xi_config * config = (cl_to_xi_config*) config_in;

    // Load the number of redshift bins
    if (c_datablock_has_value(block, config->input_section, "nbin_a")){
        status |= c_datablock_get_int(block, config->input_section, "nbin_a", &num_z_bin_A);
        status |= c_datablock_get_int(block, config->input_section, "nbin_b", &num_z_bin_B);
    }
    else{
        status |= c_datablock_get_int(block, config->input_section, "nbin", &num_z_bin_A);
        num_z_bin_B = num_z_bin_A;
    }

    if (status) {
        fprintf(stderr, "Could not load nbin in C_ell -> xi\n");
        return status;
    }
    //Also load ell array
    double * ell;

    int n_ell;
    status |= c_datablock_get_double_array_1d(block, config->input_section, "ell", &ell, &n_ell);
    if (status) {
        fprintf(stderr, "Could not load ell in C_ell -> xi\n");
        return status;
    }
    double log_ell_min = log(ell[0]);
    double log_ell_max = log(ell[n_ell-1]);
    double dlog_ell=(log_ell_max-log_ell_min)/(n_ell-1);
    error * err = NULL;
    interTable* cl_table; //= init_interTable(n_ell, log_ell_min, log_ell_max,
    //  dlog_ell, 1.0, -3.0, &err);

    char name_in[64], name_out[64];
    char xi_section[64], xi_minus_section[64];

    double log_theta_min, log_theta_max;

    // Choose the type of Hankel transform
    tpstat_t tpstat;
    if (config->filter_type == map) {
      switch(config->corr_type) {   
        case shear_shear:
          tpstat = tp_map2_poly;
          snprintf(xi_section, 64, "%s_%s", config->output_section, "map2");
          break;
        case matter:
          tpstat = tp_map2_poly;
          snprintf(xi_section, 64, "%s_%s", config->output_section, "nap2");
          break;
        case ggl:
          tpstat = tp_map2_poly;
          snprintf(xi_section, 64, "%s_%s", config->output_section, "mapnap2");
          break;
        default:
          printf("corr_type: %d\n", config->corr_type);
          printf("ERROR: Invalid corr_type %d in cl_to_xi_interface\n",config->corr_type);
          return 10;
      }
    }

    else if (config->filter_type == corrfct) {
      switch(config->corr_type) {
        case shear_shear:
          tpstat = tp_xipm;
          snprintf(xi_section, 64, "%s_%s", config->output_section, "plus");
          snprintf(xi_minus_section, 64, "%s_%s", config->output_section, "minus");
          break;
        case matter:
          tpstat = tp_w;
          strcpy(xi_section, config->output_section);
          break;
        case ggl:
          tpstat = tp_gt;
          strcpy(xi_section, config->output_section);
          break;
        default:
          printf("corr_type: %d\n", config->corr_type);
          printf("ERROR: Invalid corr_type %d in cl_to_xi_interface\n",config->corr_type);
          return 10;
      }
    }
    else {
      printf("ERROR: Invalid filter_type %d in cl_to_xi_interface\n",config->filter_type);
      return 10;
    }

    // Loop through bin combinations, loading ell,C_ell and computing xi+/-
    int j_bin_start;
    bool found_any = false;
    for (int i_bin=1; i_bin<=num_z_bin_A; i_bin++) {
        for (int j_bin=1; j_bin<=num_z_bin_B; j_bin++) {
            // read in C(l)
            double * C_ell;
            snprintf(name_in, 64, "bin_%d_%d",i_bin,j_bin);
            snprintf(name_out, 64, "bin_%d_%d",i_bin,j_bin);
            if (!c_datablock_has_value(block, config->input_section, name_in)){
                continue;
            }
            found_any=true;
            status |= c_datablock_get_double_array_1d(block, config->input_section, name_in, &C_ell, &n_ell);
            if (status) {
                fprintf(stderr, "Could not load bin %d,%d in C_ell -> xi\n", i_bin, j_bin);
                return status;
            }

            //fill cl_table for P_projected
            //need to check for zero or negative values...if there are some, can't do loglog interpolation
            int neg_vals=0;
            for (int i=0; i<n_ell; i++){
                if (C_ell[i]<=0) {
                    neg_vals += 1;
                }
            }
            //also check for zeros, and replace with v small number if all other C_ells are all +ve or -ve
            for (int i=0; i<n_ell; i++){
                if (fabs(C_ell[i])<=1.e-30) {
                    if (neg_vals==n_ell){
                        C_ell[i]=-1.e-30;
                    }
                    else if (neg_vals==0) {
                        C_ell[i]=1.e-30;
                    }
                }
            }       

            if (neg_vals == 0) {
            cl_table = init_interTable(n_ell, log_ell_min, log_ell_max,
                                   dlog_ell, 1.0, -3.0, &err);
            for (int i=0; i<n_ell; i++){
                cl_table->table[i] = log(C_ell[i]);
            }
          }
            else if (neg_vals == n_ell){
            //In this case all the C(l)s are negative, so interpolate in log(-C)
            //just remember to flip sign again at the end
            cl_table = init_interTable(n_ell, log_ell_min, log_ell_max,
                            dlog_ell, 1.0, -3.0, &err);
            for (int i=0; i<n_ell; i++){
                cl_table->table[i] = log(-C_ell[i]);
            }
            }
            else {
                static int warned=0;
            if (warned==0){
                printf("Negative values in C(l). No interpolation in log(C(l)),\n");
                printf("and no power law extrapolation. So make sure range of input ell\n");
                printf("is sufficient for this not to matter. \n");
                printf("This warning will only appear once per process. \n");
                warned=1;
            }
            cl_table = init_interTable(n_ell, log_ell_min, log_ell_max,
                             dlog_ell, 0., 0., &err);
            for (int i=0; i<n_ell; i++){
                cl_table->table[i] = C_ell[i];
            }
            }

        p_projected_data d;
        d.cl_table = cl_table;
        //d.f = f;
        if (neg_vals == 0) {
          tpstat_via_hankel(&d, xi, &log_theta_min, &log_theta_max, tpstat, &P_projected_loglog, i_bin, j_bin, &err);
        }
        else if (neg_vals == n_ell) {
          tpstat_via_hankel(&d, xi, &log_theta_min, &log_theta_max, tpstat, &P_projected_loglog, i_bin, j_bin, &err);
          double xip_orig,xim_orig;
          for (int i=0; i<N_thetaH; i++){
            xip_orig = xi[0][i];
            xi[0][i] =-1*xip_orig;
            if ((config->filter_type == corrfct)&&(config->corr_type == shear_shear)) {
              xim_orig = xi[1][i];
              xi[1][i] =-1*xim_orig;
            }
          }
        }
        else {
          tpstat_via_hankel(&d, xi, &log_theta_min, &log_theta_max, tpstat, &P_projected_logl, i_bin, j_bin, &err);
        }

        // Save to th block
        if ((config->filter_type == corrfct)&&(config->corr_type == shear_shear)) {
            status |= c_datablock_put_double_array_1d(block, xi_section, name_out, xi[0], N_thetaH);
            status |= c_datablock_put_double_array_1d(block, xi_minus_section, name_out, xi[1], N_thetaH);
        }
        else {
            status |= c_datablock_put_double_array_1d(block, xi_section, name_out,
                      xi[0], N_thetaH);
        }
        free(C_ell);
        del_interTable(&d.cl_table);
        }
    }
    if (!found_any){
        fprintf(stderr, "WARNING: I COULD NOT FIND ANY SPECTRA OF THE FORM \n");
        fprintf(stderr, "xiplus_i_j, ximinus_i_j, wmatter_i_j, or tanshear_i_j.\n");
        fprintf(stderr, "THIS IS ALMOST CERTAINLY AN ERROR.\n");
        status = 1;
    }
    // Save theta values...check these are being calculated correctly at some point
    double dlog_theta= (log_theta_max-log_theta_min)/((double)N_thetaH-1.0);
    double logtheta_center = 0.5*(log_theta_max+log_theta_min);
    int nc = N_thetaH/2+1;
    double *theta_vals = malloc(sizeof(double)*N_thetaH);
    for (int i=0; i<N_thetaH; i++){
        theta_vals[i] = exp(log_theta_min+i*dlog_theta);
    }
    status |= c_datablock_put_double_array_1d(block, xi_section, "theta",
                  theta_vals, N_thetaH);
    //Include units
    status |= c_datablock_put_metadata(block, xi_section, "theta", "unit", "radians");
    status |= c_datablock_put_string(block, xi_section, "sep_name", "theta");

    //copy over metadata from cl_section
    status |= copy_metadata(status, block, xi_section,
                  config->input_section );

    if ((config->filter_type == corrfct)&&(config->corr_type == shear_shear)) {
        status |= c_datablock_put_double_array_1d(block, xi_minus_section, "theta",
                      theta_vals, N_thetaH);
        //Include units
        status |= c_datablock_put_metadata(block, xi_minus_section, "theta", "unit", "radians");
        status |= copy_metadata(status, block, xi_minus_section,
                  config->input_section );
        status |= c_datablock_put_string(block, xi_minus_section, "sep_name", "theta");
    }

    //Clean up

    for (int i=0; i<count; i++) free(xi[i]);
    free(xi);
    free(ell);
    free(theta_vals);

    return status;
}

int cleanup(void * config_in)
{
	// Free the memory that we allocated in the setup
	cl_to_xi_config * config = (cl_to_xi_config*) config_in;
	free(config);
  return 0;
}
