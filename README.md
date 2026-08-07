# Biostream analysis notebook
Python code for extracting and  training models on bispecific antibody data.

## Variables to be set in main() 
String variable with the path of the .csv file with the data sequences

filepath =

Dictionary with keys the names of the columns in the .csv file to be used as a Targets. Values of the keys are the cutoff thresholds to applied on the Target during diagnostic plots.

**targets_to_test =**

List of strings with names of the methods to train models with. PLSRegression, SVR, Random Forest, ElasticNet are supported.

**models_to_test =**

String of desired transformation to be applied on the Target distribution. Options include: None, 'log1p', 'box-cox', 'yeo-johnson', or 'weighted-yeo-johnson'

**transform_strategy =**

A list of models to extract sets of features. Supported options "facebook/esm2_t6_8M_UR50D", "facebook/esm2_t33_650M_UR50D"

**esm_model_selections =**

String with the name of the column in the .csv file to color code the diagnostic plots

**antibody_format_column =**

Name of the column in the .csv file based on which the script will split the dataset to plot generalization plots

**out_of_group_split_column =**

Boolean variables that will decide if propermab or cqa features will be included in the analysis.

**use_external_features =**

**use_cqa_features =**

 If using a transformation of Target, specify the custom weight column here

 **weighting_column =**

 
