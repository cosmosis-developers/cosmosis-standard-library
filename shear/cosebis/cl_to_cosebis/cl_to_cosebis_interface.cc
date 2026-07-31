///for c++ use the .hh and for c the .h version
//This deals with the inputs and outputs
#include "datablock/datablock.hh"
//This is just a header file which defines the different section names
#include "datablock/section_names.h"
#include <typeinfo>

/*CosmoSIS interface file for going from shear C(l) to E/B - COSEBIs
*/
#include "COSEBIs.h"

//could add an option to read B-modes in and use it to constrain c-term

extern "C" {

	const string shear_cl = SHEAR_CL_SECTION;

	typedef struct COSEBIs_config 
	{
		string input_section_name; //input section name the default is: shear_cl
		string output_section_name;// where cosebis results are saved, default: cosebis
		int n_max; // largest n-mode to be caluclated for COSEBIs: 1,2,...,n_max
		number theta_min; //theta_min
		number theta_max; //theta_max

		COSEBIs *cosebis; //cosebis object

	} COSEBIs_config;

	///define a structure with everything that is needed to be read in setup and sent to execute
	//Some of these are read from the ini file. For example input_section_name and n_max
	//Some are initialized in the setup, such as cosebis. COSEBIs is a class that produces En/Bn and 
	//their covariance, etc. 
	int get_option(cosmosis::DataBlock * options, const string &name, string &parameter)
	{
	
		auto status = options->get_val(OPTION_SECTION, name, parameter);
		if (status!=DBS_SUCCESS) 
		{
			parameter = "";
			cerr<< "I could not find or understand the parameter in the cosebis section: " 
				<< name << std::endl; 
			return 1;
		}
		return 0;
	}
	
	void * setup(cosmosis::DataBlock * options, cosmosis::DataBlock * block)
	{
		//options reads the ini file
		//define config here and then read from options the relevant input quantities
		COSEBIs_config * config = new COSEBIs_config;

  		string sectionName=OPTION_SECTION;

		DATABLOCK_STATUS status = (DATABLOCK_STATUS)0;
		const DATABLOCK_STATUS failure = (DATABLOCK_STATUS)1;

		clog<<endl<<endl;
		clog<<"********* COSEBIs interface setup *********"<<endl;
		clog<<endl;

		//get input section name, default= shear_cl
		status=options->get_val<string>(sectionName, string("input_section_name"), config->input_section_name);
		if (status) 
		{
			clog<<"You did not define the input_section_name,";
			clog<<"setting to the default: shear_cl"<<endl;
			config->input_section_name=string("shear_cl");
		}
		else
			clog<<"The input_section_name:"<<config->input_section_name<<endl;

		//get output_section_name, default= "cosebis"
		status=options->get_val<string>(sectionName, string("output_section_name"), config->output_section_name);
		if (status) 
		{
			clog<<"You did not define the out_section_name, ";
			clog<<"setting to the default: cosebis"<<endl;
			config->output_section_name=string("cosebis");
		}
		else
			clog<<"The output_section_name:"<<config->output_section_name<<endl;

		//get theta_min value, default=0.5
		status=options->get_val<number>(sectionName, string("theta_min"), config->theta_min);
		if (status) 
		{
			clog<<"You did not define the COSEBIs range theta_min,"<<endl;
			clog<<"setting to the default value of"<<0.5<<endl;
			config->theta_min=0.5;
		}
		else
			clog<<"COSEBIs range theta_min="<<config->theta_min<<endl;

		//get theta_max value, default=300
		status=options->get_val<number>(sectionName, string("theta_max"), config->theta_max);
		if (status) 
		{
			clog<<"You did not define the COSEBIs range theta_max,"<<endl;
			clog<<"setting to the default value of"<<300.<<endl;
			config->theta_max=300.;
		}
		else
			clog<<"COSEBIs range theta_max="<<config->theta_max<<endl;

		//get n_max value, default=10
		status=options->get_val<int>(sectionName, string("n_max"), config->n_max);
		if (status) 
		{
			clog<<"You did not define the number of COSEBIs modes that you require, n_max,"<<endl;
			clog<<"setting to the default value of"<<5<<endl;
			config->n_max=5;
		}
		else
			clog<<"Calculating COSEBI nodes up to n_max="<<config->n_max<<endl;

		//get Wn, Tn and output Tn folder names
		string WnFolderName,TnFolderName,OutputTnFolderName,WnFileName;
		WnFolderName = COSEBIS_DIR  "WnLog/";
		TnFolderName = COSEBIS_DIR  "TLogsRootsAndNorms/";
		OutputTnFolderName= COSEBIS_DIR  "TpnLog/";  // This is not needed for cl_to_cosebis and is a dummy variable
		WnFileName   = "WnLog";

		// These values were used in the KiDS-1000 analysis
		// int precision = 20;
		// int Nlbins    = 1000000;

		// If you only want to only use mode up to n=5, however, these are sufficient and faster to calculate
		int precision = 10;
		int Nlbins    = 100000;

		status=options->get_val<string>(sectionName, string("Wn_Output_FolderName"), WnFolderName);
		if(status)
		{
			clog<<"You did not define the name for the WnLog folder Wn_Output_FolderName, ";
			clog<<"setting to the default: "<<WnFolderName<<endl;
		}
		else
			clog<<"The Wn_Output_FolderName:"<<WnFolderName<<endl;

		status=options->get_val<string>(sectionName, string("Wn_file_name"), WnFileName);
		if(status)
		{
			clog<<"You did not define the Wn_file_name, ";
			clog<<"setting to the default: "<<WnFileName<<endl;
		}
		else
			clog<<"The Wn_file_name:"<<WnFileName<<endl;

		status=options->get_val<int>(sectionName, string("table_precision"), precision);
		if(status)
		{
			clog<<"As you did not set the table_precision, ";
			clog<<"I will set this to the default: "<<precision<<endl;
		}
		else
			clog<<"The table_precision:"<<precision<<endl;

		status=options->get_val<int>(sectionName, string("number_of_Wn_l_bins"), Nlbins);
		if(status)
		{
			clog<<"As you did not set the number_of_Wn_l_bins, ";
			clog<<"I will set this to the default: "<<Nlbins<<endl;
		}
		else
			clog<<"The number_of_Wn_l_bins:"<<Nlbins<<endl;


		status=options->get_val<string>(sectionName, string("Roots_n_Norms_FolderName"), TnFolderName);
		if(status)
		{
			clog<<"You did not define the name of the Root and Norms folder, Roots_n_Norms_FolderName, "; 
			clog<<"setting to the default: "<<TnFolderName<<endl;
		}
		else
			clog<<"The Roots_n_Norms_FolderName:"<<TnFolderName<<endl;


		//   initialize COSEBIs
		COSEBIs *cosebis = new COSEBIs();
		cosebis->initialize(config->n_max,config->theta_min,config->theta_max,1 //npair set to one for now, will be set seperately in execute to the correct value
				,WnFolderName,TnFolderName,OutputTnFolderName,WnFileName,precision);

		cosebis->setWns(config->n_max,Nlbins);
		config->cosebis=cosebis;
		return (void *) config;
		// config is sent to execute 
	}

	DATABLOCK_STATUS execute(cosmosis::DataBlock *block, void *config_in) 
	{
		enable_gsl_error_handling();
		// Config is whatever you returned from setup above
		// Block is the collection of parameters and calculations for
		// this set of cosmological parameters
		COSEBIs_config *config= (COSEBIs_config*) config_in;
		DATABLOCK_STATUS status = (DATABLOCK_STATUS)0;
		const DATABLOCK_STATUS failure = (DATABLOCK_STATUS)1;

		//lets save some of the atributes to block
		status = block->put_val<double>(config->output_section_name, string("theta_max"), config->theta_max);
		status = block->put_val<double>(config->output_section_name, string("theta_min"), config->theta_min);

		//get the number of redshift bins from cosmosis
		int num_z_bin_A;
		int num_z_bin_B;
		status = block->get_val(config->input_section_name, string("nbin_a"), num_z_bin_A);
		if(status)
		{
			status = block->get_val(config->input_section_name, string("nbin"), num_z_bin_A);
			if(status)
			{
				clog<<"looked for nbin_a first and couldn't find it. Then looked for nbin and it wasn't there"<<endl;
				return status;
			}
			num_z_bin_B = num_z_bin_A;
		}
		else
		{
			status = block->get_val(config->input_section_name, string("nbin_b"), num_z_bin_B);
			if(status)
			{
				clog<<"looked for nbin_b it is not set"<<endl;
				return status;
			}
		}

		//get is_auto
		bool is_auto;

		status = block->get_val(config->input_section_name, string("is_auto"), is_auto);
		if (status)
		{
			clog << "WARNING: could not find the value of is_auto. Setting it to True." << endl;
			is_auto = true;
		}

		vector<number> ell,logell;
		//get ell vector
		status = block->get_val(config->input_section_name, string("ell"), ell);
		if (status) 
		{
			clog<<"Could not load ell in C_ell to COSEBIs"<<endl;
			return status;
		}

		int nell=ell.size();
		//make logell to send to COSEBIs
		for(int i=0; i<nell; i++)
		{
			logell.push_back(log(ell[i]));
		}

		int n_max=config->n_max;
		vector<int> n_vals(n_max);
		for(int n=0;n<n_max;n++)
		 	n_vals[n]=n+1;

		status = block->put_val<vector<int> >(config->output_section_name, string("n"), n_vals);


		//here is where we read in the Cls and calculate COSEBIs
		int start_ind=0;
		int start_ind_2D=0;
		for (int i_bin=1; i_bin<=num_z_bin_A; i_bin++) 
		{
			for (int j_bin=1; j_bin<=num_z_bin_B; j_bin++) 
			{
				// read in C(l)
				vector<number> C_ell;
				string name_in=string("bin_")+toString(j_bin)+string("_")+toString(i_bin);
				//check if C(l) exists for this z-bin combination
				bool has_val = block->has_val(config->input_section_name, name_in);
				if (has_val) 
				{
					//if C(l) exists then read in
					status = block->get_val<vector<number> >(config->input_section_name, name_in, C_ell);
					
					matrix En_mat;

					//this also extrapolates the input Cl to LHIGH
					config->cosebis->setPower_single_withExtrapolation(logell,C_ell);

					//Calculate COSEBIs for the given C(l)
					En_mat=config->cosebis->calEn();

					// turning the matrix into a vector to be saved in block
					vector<number> En_vec(En_mat.rows);
					for(int m=0 ;m<En_mat.rows ;m++)
					{
						En_vec[m]=En_mat.get(m);
					}
					status = block->put_val<vector<number> >(config->output_section_name, name_in, En_vec);

				}
			}
		}
		status = block->put_val<double>(config->output_section_name, string("nbin_a"), num_z_bin_A);
		status = block->put_val<double>(config->output_section_name, string("nbin_b"), num_z_bin_B);
		status = block->put_val(config->output_section_name, "theta_min", config->theta_min);
    	status = block->put_val(config->output_section_name, "theta_max", config->theta_max);
    	status = block->put_val(config->output_section_name, "is_auto", is_auto);
		status = block->put_val(config->output_section_name, "input_section_name", config->input_section_name);

		
		disable_gsl_error_handling();
		return status;
	}


	void cleanup(void * config)
	{
		// This function is called once at the end of the program
		// and is typically used to clean up memory or write logs/files
		COSEBIs_config *config_ = (COSEBIs_config*) config;
		delete config_->cosebis;
		delete config_;
	}
}// end of extern C



    
