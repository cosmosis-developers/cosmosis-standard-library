#include "clik.h"
#include "datablock/c_datablock.h"
#include <math.h>

// Hard-coded maximum number of likelihoods.
// This is far more than have been released.
// I'm just doing this to avoid some mallocs and stuff.
#define MAX_NUMBER_LIKES 128

typedef struct configuration_data{
	int ndata, nlensing;
	bool save_separate_likelihoods;
	clik_object * clik_data[MAX_NUMBER_LIKES];
	clik_lensing_object * clik_lensing_data[MAX_NUMBER_LIKES];

} configuration_data;

typedef enum {clik_type_standard, clik_type_lensing} clik_type;




static
int read_requested_files(configuration_data * config, c_datablock * options, clik_type mode){
	int status = 0;
	char param[512];
	error * err = initError();
	char * prefix;
	switch (mode) {
	case clik_type_standard:
		prefix = "data";
		break;
	case clik_type_lensing:
		prefix = "lensing";
	  break;
	default:
	  fprintf(stderr, "Unknown type of clik file - this is a coding error, please report it.\n");
	  exit(1);
	  break;
	}	


	int n=0;
	while (1){
		if (n==127){
			fprintf(stderr, "Somehow you have reached the hard-coded maximum number of clik files supported by cosmosis: %d.", MAX_NUMBER_LIKES);
			fprintf(stderr, "Probably you are just messing about trying to break things.  Well done, then.\n");
			fprintf(stderr, "If not (somehow) you will need to modify MAX_NUMBER_LIKES in cosmosis-stadard-library/likelihood/planck2014/planck_interface.c\n");
			exit(1);
		}
		snprintf(param, 128, "%s_%d", prefix, n+1);
		char * value;
		status |= c_datablock_get_string_default(options, OPTION_SECTION, 
			param, "", &value);

		if (strlen(value)==0) break;

		printf("Looking for clik Planck likelihood file %d: %s\n", 
			n+1, value);

		switch (mode) {
			case clik_type_standard:
				config->clik_data[n] = clik_init(value, &err);
				break;
			case clik_type_lensing:
				config->clik_lensing_data[n] = clik_lensing_init(value, &err);
			  break;
			default:
			  fprintf(stderr, "Unknown type of clik file - this is a coding error, please report it.\n");
			  exit(1);
			  break;
		}
		n++;


		if (isError(err)){
			fprintf(stderr,"There was an error initializating one of the Planck likelihoods.\n");
			fprintf(stderr,"Here is the error message:\n");
			printError(stderr,err);
			fprintf(stderr,"\nIf it mentions \"stat\" then this probably means that a likelihood file was not found.\n\n");
			fprintf(stderr,"You can run the script examples/get-planck-data.sh to download Planck 2018 data.\n\n");
			exit(1);
		}
	}
	endError(&err);
	return n;
}


configuration_data * setup(c_datablock * options){
	configuration_data * config = malloc(sizeof(configuration_data));

	int status = 0;
	config->ndata=read_requested_files(config, options, clik_type_standard);
	config->nlensing=read_requested_files(config, options, clik_type_lensing);

	if ((config->ndata + config->nlensing)==0){
		fprintf(stderr, "No data files were specified for Planck at all!\n");
		fprintf(stderr, "In the options file (not the values.ini file) you need to set in the section for Planck at least the first of:\n");
		fprintf(stderr, "   data_1 = ...\n");
		fprintf(stderr, "   data_2 = ...\n");
		fprintf(stderr, "   data_3 = ...\n");
		fprintf(stderr, "   data_4 = ...\n");
		fprintf(stderr, "   etc. and/or\n");
		fprintf(stderr, "   lensing_1 = ...\n");
		fprintf(stderr, "   lensing_2 = ...\n");
		fprintf(stderr, "   etc.\n");
		fprintf(stderr, "The Planck data files can be downloaded from:\n");
		fprintf(stderr, "http://pla.esac.esa.int/pla/aio/planckProducts.html\n");
		fprintf(stderr, "or some are packaged with the planck2015 cosmosis module.\n");
		exit(1);
	}

	status |= c_datablock_get_bool_default(options, OPTION_SECTION, "save_separate_likelihoods", false,
		&(config->save_separate_likelihoods));

	if (status){
		fprintf(stderr, "Error reading save_separate_likelihoods parameter (boolean) in planck 2015\n");
		exit(1);
	}

	if (config->save_separate_likelihoods){
		printf("Will save each Planck likelihood separately (per file) to the block, as well as their sum\n");
		printf("Note that in this case you must explicitly specify planck2018 in the likelihoods \n");
		printf("list in the parameter file; you can't just leave it blank for cosmosis to determine,\n");
		printf("because it will be wrong.\n");
	}
	else{
		printf("Will save each a single total likelihood for all Planck likelihood files\n");
	}


	// typedef struct configuration_data{
	// 	int ndata, nlensing;
	// 	bool save_separate_likelihoods;
	// 	clik_object * clik_data[MAX_NUMBER_LIKES];
	// 	clik_lensing_object * clik_lensing_data[MAX_NUMBER_LIKES];
	
	// } configuration_data;
	

	// Print required nuisance parameters
	printf("Required nuisance parameters (may repeat!):\n");
	for (int i=0;i<config->ndata;i++){
		int n_nuisance;
		parname * nuisance_names;
		n_nuisance = clik_get_extra_parameter_names(config->clik_data[i], &nuisance_names, NULL);
		for (int j=0; j<n_nuisance; j++){
			printf("    - %s\n", nuisance_names[j]);
		}
	}
	// same with lensing
	printf("And for lensing:\n");
	for (int i=0;i<config->nlensing;i++){
		int n_nuisance;
		parname * nuisance_names;
		n_nuisance = clik_lensing_get_extra_parameter_names(config->clik_lensing_data[i], &nuisance_names, NULL);
		for (int j=0; j<n_nuisance; j++){
			printf("    - %s\n", nuisance_names[j]);
		}
	}

	return config;
}


static 
int extract_c_ell(c_datablock * block, int n_cl, 
	char * package_names[n_cl], int lmax_by_type[n_cl], int n_nuisance, 
	parname nuisance_names[n_nuisance], double *p, double *cl)
{
	int i;
	int status=0;

	for (i=0;i<n_cl;i++) {
		// Check lmax for this spectrum.  If not needed, continue
		int lmax = lmax_by_type[i];

		if (lmax<=0) continue;
		// Otherwise fill in first the trivial cl and then the rest, loaded fromt the package
		cl[0]=0.0;
		cl[1]=0.0;
		int nread=0;
		double * ell_squared_cl = NULL;
		status |= c_datablock_get_double_array_1d(block, CMB_CL_SECTION, package_names[i], 
			&ell_squared_cl, &nread);

		if (status){
			fprintf(stderr, "Could not get C_ell values for %s (status=%d)\n", 
				package_names[i],status);
			status = 1;
			return status;
		}
		if (nread<lmax-1){
			fprintf(stderr, "Could not get enough C_ell values for %s (read lmax=%d but needed lmax=%d).  May need to adjust lmax.\n", 
				package_names[i], nread+1, lmax_by_type[i]);
			status = 1;
			return status;
		}


		// Transform to raw cl from l**2 cl
		int ell;
		for (ell=2; ell<=lmax; ell++) {
			cl[ell] = ell_squared_cl[ell-2] / (ell*(ell+1.0)/2.0/M_PI);
		}
		free(ell_squared_cl);

		// Move the spectrum pointer forward ready for the next spectrum (e.g. TT->EE)
		cl+=lmax+1;
	}

	// ... and now get the nuisance parameters
	for (i=0; i<n_nuisance; i++){
		int this_status=c_datablock_get_double(block, PLANCK_SECTION, nuisance_names[i], 	cl+i  ); //ok
		// Report any missing parameters
		if (this_status) fprintf(stderr, "Could not get required Planck parameter %s in section %s\n",nuisance_names[i], PLANCK_SECTION);
		status |= this_status;
	}
	return status;
}


static 
int run_clik_cosmosis(c_datablock * block, clik_object * like_obj, double * output_like)
{
	int i;
	const int N_CL = 6;
	int param_count=0;
	int status = 0;
	char * package_names[] = {"TT","EE","BB","TE","TB","EB"};
	int lmax_by_type[N_CL];

	// Planck internal error system
	error * err = initError();

	// Get the lmax values for the different spectra and sum them
	clik_get_lmax(like_obj, lmax_by_type,&err);
	for (i=0;i<N_CL;i++) if (lmax_by_type[i]>0) param_count += lmax_by_type[i]+1; 

	// Check for any errors from the Planck library
	if (isError(err)){
		fprintf(stderr,"There was an error getting lmax from a Planck likelihood.\n");
		endError(&err);
		return 1;
	}

	// Get the number and names of extra parameters and add to total
	int n_nuisance;
	parname * nuisance_names;
	n_nuisance = clik_get_extra_parameter_names(like_obj, &nuisance_names, &err);
	param_count += n_nuisance;

	// Check for any errors from the Planck library
	if (isError(err)){
		fprintf(stderr,"There was an error getting nuisance params from a Planck likelihood.\n");
		endError(&err);
		return 2;
	}

	// p is the overall space and cl is the space within it for each spectra.
	double * p = malloc(sizeof(double)*param_count);
	double * cl = p;

	// Get the c_ell we need, and the nuisance parameters (if any)
	status |= extract_c_ell(block, N_CL, package_names, lmax_by_type, n_nuisance, 
		nuisance_names, p,cl);

	free(nuisance_names);
	// Check for any errors reading things in
	if (status){
		free(p);
		return status;
	}

	// Compute the actual likelihood
	double like = clik_compute(like_obj, p, &err);
	free(p);
	// ...  and check for errors
	if (isError(err)){
		status = 1;
		fprintf(stderr,"Planck likelihood error:\n");
		printError(stderr, err);
	}
	// Clean up Planck error system and 
	// return required outputs
	endError(&err);	
	*output_like = like;
	return status;
}


static 
int run_clik_cosmosis_lensing(c_datablock * block, clik_object * like_obj, double * output_like)
{
	int i;
	const int N_CL=7;
	int param_count=0;
	int status = 0;
	char * package_names[] = {"PP","TT","EE","BB","TE", "TB", "EB"};
	error * err = initError();

	int lmax_by_type[N_CL];


	// Get the lmax values for the different spectra and sum them
	clik_lensing_get_lmaxs(like_obj, lmax_by_type,&err);
	for (i=0;i<N_CL;i++) if (lmax_by_type[i]>0) param_count += lmax_by_type[i]+1; 
	// Check for any errors from the Planck library
	if (isError(err)){
		fprintf(stderr,"There was an error getting lmax from a Planck lensing likelihood:\n");
		printError(stderr,err);
		return 1;
	}

	// Get the number and names of extra parameters and add to total
	int n_nuisance;
	parname * nuisance_names;
	n_nuisance = clik_lensing_get_extra_parameter_names(like_obj, &nuisance_names, &err);
	param_count += n_nuisance;

	// Check for any errors from the Planck library
	if (isError(err)){
		fprintf(stderr,"There was an error getting nuisance params from a Planck lensing likelihood.\n");
		printError(stderr,err);
		return 2;
	}

	// p is the overall space and cl is the space within it for each spectra.
	double * p = malloc(sizeof(double)*param_count);
	double * cl = p;

	// Get the c_ell we need, and the nuisance parameters (if any)
	status |= extract_c_ell(block, N_CL, package_names, lmax_by_type, n_nuisance, 
		nuisance_names, p,cl);

	free(nuisance_names);

	// Check for any errors reading things in and quit early if problem
	if (status){
		free(p);
		return status;
	}


	// Compute the actual likelihood and check for errors
	double like = clik_lensing_compute(like_obj, p, &err);
	free(p);

	//Clean up
	if (isError(err)){
		status = 1;
		fprintf(stderr, "Error running Planck lensing\n");
	}

	// Clean up Planck error system and 
	// return required outputs
	endError(&err);	
	*output_like = like;

	return status;
}






int execute(c_datablock * block, configuration_data * config){

	double like = 0.0;
	int status = 0;

	for (int i=0; i<config->ndata; i++){
		// Compute the likelihood for this file.
		double like_i = 0.0;
		status |= run_clik_cosmosis(block, config->clik_data[i], &like_i);

		//
		if (config->save_separate_likelihoods){
			char name[64];
			snprintf(name, 64, "PLANCK18_LIKE_%d", i+1);
			status |= c_datablock_put_double(block, LIKELIHOODS_SECTION, name, like_i);
		}

		like += like_i;
	}

	if (status) return status;

	for (int i=0; i<config->nlensing; i++){
		// Compute the likelihood for this file.
		double like_i = 0.0;
		status |= run_clik_cosmosis_lensing(block, config->clik_lensing_data[i], &like_i);

		//
		char name[64];
		snprintf(name, 64, "PLANCK18_LENSING_LIKE_%d", i+1);
		status |= c_datablock_put_double(block, LIKELIHOODS_SECTION, name, like_i);

		like += like_i;
	}


	if (status) return status;

	// Save the total likelihood
	status |= c_datablock_put_double(block, LIKELIHOODS_SECTION, "PLANCK2018_LIKE", like);

	return status;
}
