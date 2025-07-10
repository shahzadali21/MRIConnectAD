# module: LCDP_preprocessing.py
# -*- coding: utf-8 -*-
# Author: Shahzad Ali
"""
This script handles the preprocessing of clinical data.
It reads the data, performs basic EDA, extracts features and targets,
scales numeric features for training and test sets separately for all classification types,
and saves the final processed data to the specified output directory structure.
"""


import os
import logging
import argparse
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from LCDP_model_utils import save_data


# General Preprocessing to prepare reusable dataset
def general_preprocessing(df, output_dir):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    logging.info(f"\n\n######################## RAW DATA EDA #####################")
    logging.info(f"Shape of dataframe: {df.shape}, \nValue counts of DX: {df['DX'].value_counts()}, \nValue counts of Research Group: {df['RG'].value_counts()}")
    logging.info(f"SETUP Information: \nDataframe: {df.head()}")
    # Save the raw dataframe
    df.to_csv(os.path.join(output_dir, '0_Raw_data.csv'), index=False)
    logging.info(f"Saved preprocessed data to {output_dir}")

    logging.info(f"\n\n######################## REMOVING UNNECESSARY COLUMNS #####################")
    unnecessary_cols = ['RID', 'COLPROT', 'ORIGPROT', 'SITE', 'Final_Status', 'DX_bl', 'Transition_DXbl_DX', 'Transition_DXbl_Group', 'EXAMDATE_bl', 'EXAMDATE', 'AGE_bl']
    df.drop(columns=unnecessary_cols, inplace=True)
    logging.info(f"Removed unnecessary columns: {unnecessary_cols}, \nShape after removing unnecessary_cols: {df.shape}, \nColumns in df after removing unnecessary_cols: {list(df.columns)}")


    logging.info(f"\n\n############################## LABEL ENCODING ############################")
    # Optional - Convert columns to 'category' type (considering 'APOE4' as cateogory)
    columns_to_convert = ['APOE4']     # define list of columns to consider as category
    df[columns_to_convert] = df[columns_to_convert].astype('category')
    
    categorical_cols = df.select_dtypes(exclude=['int64', 'float64']).columns.tolist()   # df.select_dtypes(include=['object', 'category']).columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Log the categories and unique values before encoding
    for column in categorical_cols:
        if column == 'Subject ID':
            continue
        unique_values = df[column].unique()
        print(f"Unique values in {column}: {unique_values}")

    logging.info(f"List of Categorical Columns: {categorical_cols}, \nList of Numerical Columns: {numerical_cols}")
    
    # Manually define the mapping for label encoding based on desired order
    sex_map = {'F': 0, 'M': 1}
    marry_map = {'Never married': 0, 'Married': 1, 'Widowed': 2, 'Divorced': 3, 'Unknown': 4}
    rg_dx_map = {'CN': 0, 'MCI': 1, 'AD': 2}
    csf_map = {'A-': 0, 'A+': 1}

    # Apply mappings: perform label encoding and insert new columns right after the original columns in one go
    df.insert(df.columns.get_loc('Sex') + 1, 'Gender', df['Sex'].map(sex_map))
    df.insert(df.columns.get_loc('PTMARRY') + 1, 'Marital_status', df['PTMARRY'].map(marry_map))
    df.insert(df.columns.get_loc('RG') + 1, 'RG_le', df['RG'].map(rg_dx_map))
    df.insert(df.columns.get_loc('DX') + 1, 'DX_le', df['DX'].map(rg_dx_map))
    df.insert(df.columns.get_loc('Amy_Status') + 1, 'Amyloid_Status', df['Amy_Status'].map(csf_map))

    df.rename(columns={'PTEDUCAT': 'Education_level'}, inplace=True)
    logging.info(f"\nShape of df after Label Encoding: {df.shape}, \nColumns in df after Label Encoding: {list(df.columns)}")

    logging.info(f"\n\n######################## Basic Preprocessed DATA EDA #####################")
    logging.info(f"Shape of dataframe: {df.shape}, \nValue counts of DX: {df['DX'].value_counts()}, \nValue counts of Research Group: {df['RG'].value_counts()}")
    logging.info(f"SETUP Information: \nDataframe: {df.head()}")

    # Save the preprocessed dataframe
    df.to_csv(os.path.join(output_dir, '1_Selected_Data.csv'), index=False)
    logging.info(f"Saved preprocessed data to {output_dir}")

# Feature-Specific Processing for Each Combination
def regression_specific_processing(preprocessed_file, target_column, feature_combination_name, selected_features, output_dir, scaler_type, test_size=0.2, seed=42, use_smote=False):
    """
    Processes data for regression tasks with two subsets:
    - Subset 1: Without NaN values in the target column.
    - Subset 2: Without NaN and zero values in the target column.
    """
    df = pd.read_csv(preprocessed_file)

    # Ensure target column exists
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in the dataset.")

    # Create subsets
    subset_1 = df[df[target_column].notna()]  # Without NaN values
    subset_2 = subset_1[subset_1[target_column] != 0]  # Without NaN and zero values

    #subsets = {"without_nan": subset_1, "without_nan_zero": subset_2}
    subsets = {"without_nan": subset_1}

    for subset_name, subset_df in subsets.items():
        # Create directory for the subset and feature combination
        subset_dir = create_directory_structure(output_dir, feature_combination_name, subset_name)


        # Filter columns based on the selected features and target
        selected_columns = list(selected_features) + [target_column]
        subset_df = subset_df[selected_columns]
        logging.info(f"Shape of dataframe: {subset_df.shape}, \nFiltered columns for processing: {selected_columns}, \nValue counts of {target_column}: {df[target_column].value_counts()}")
        # Save the final preprocessed data for debugging and analysis
        save_data(subset_df, os.path.join(output_dir, f'2_Preprocessed_final_{feature_combination_name}.csv'))

        logging.info(f"\n\n##################### TRAIN-TEST SPLIT and STANDARDIZATION ####################")
        # Split features and target
        y = subset_df[target_column]
        X = subset_df.drop(columns=[target_column])
        logging.info(f"Shape of Features: {X.shape}, \nShape of Target: {y.shape}")

        # Split the data
        logging.info(f"Splitting {subset_name}, Feature Combination: {feature_combination_name}, into training and test sets with test size {test_size} and seed {seed}")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
        logging.info(f"Split complete. X_train shape: {X_train.shape}, X_test shape: {X_test.shape}, y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")
         
        # Apply SMOTE (if yes, pass as parameter)
        if use_smote:
            smote = SMOTE(random_state=seed)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            logging.info(f"Applied SMOTE. Class distribution in training data: \n{y_train.value_counts()}")

        # Standardize numerical features
        numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
        scaler = MinMaxScaler() if scaler_type == "minmax" else StandardScaler()
        X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
        X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
        logging.info(f"Standardized numeric features using {scaler_type} scaler")

        # Save processed data
        save_data(X_train, os.path.join(subset_dir, "data", "X_train.csv"))
        save_data(X_test, os.path.join(subset_dir, "data", "X_test.csv"))
        save_data(y_train, os.path.join(subset_dir, "data", "y_train.csv"))
        save_data(y_test, os.path.join(subset_dir, "data", "y_test.csv"))
        print(f"Saved processed data for {subset_name}, Feature Combination: {feature_combination_name}")

    
    
# Create directory structure for feature combinations and comparisons
def create_directory_structure(output_dir, feature_combination_name, subset_name):
    feature_combination_dir = os.path.join(output_dir, feature_combination_name)
    os.makedirs(feature_combination_dir, exist_ok=True)

    # Create subdirectory for each subset within the feature combination directory
    subset_dir = os.path.join(feature_combination_dir, subset_name)
    os.makedirs(os.path.join(subset_dir, 'data'), exist_ok=True)
    os.makedirs(os.path.join(subset_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(subset_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(subset_dir, 'results'), exist_ok=True)

    return subset_dir    
    

# Define feature sets - (Demographic | Clinical | GT (Local/Global) | MO | MS Features)
demographic_features = ['Age', 'Education_level', 'Gender', 'Marital_status', 'APOE4']  # 'RG', 'DX', 'deltaMMSE/Years' 'Sex', 'PTMARRY',|| 'Sex_M', 'PTMARRY_Widowed', 'PTMARRY_Never married', 'PTMARRY_Married', 'PTMARRY_Unknown' (only if they are one-hot encoded)
clinical_features = ['MMSE_bl', 'MMSE', 'MOCA_bl', 'MOCA', 'CDRSB_bl', 'CDRSB', 'ADAS11_bl', 'ADAS11', 'ADAS13_bl', 'ADAS13', 'ADASQ4_bl', 'ADASQ4']
GT_local_metrics = [f'degree_centrality_node_{i}' for i in range(82)] + [f'clustering_coefficient_node_{i}' for i in range(82)] + [f'betweenness_centrality_node_{i}' for i in range(82)] + [f'eigenvector_centrality_node_{i}' for i in range(82)] + [f'closeness_centrality_node_{i}' for i in range(82)] + [f'node_strength_node_{i}' for i in range(82)] + [f'pagerank_node_{i}' for i in range(82)]
GT_global_metrics = ['density', 'modularity', 'assortativity', 'transitivity', 'global_efficiency', 'characteristic_path_length', 'diameter', 'degree_distribution_entropy', 'resilience', 'spectral_radius', 'small_worldness', 'avg_clustering_coefficient', 'avg_degree', 'avg_betweenness_centrality', 'avg_edge_betweenness_centrality', 'avg_eigenvector_centrality', 'avg_closeness_centrality', 'avg_node_strength', 'avg_pagerank']
microstructural_features = ['mean_MD', 'mean_FA', 'TBSS_WMmaskFA', 'LH_meanMD', 'RH_meanMD']
morphometric_features = ['BPV', 'Left-Thalamus', 'Left-Caudate', 'Left-Putamen', 'Left-Pallidum', 'Left-Hippocampus', 'Left-Amygdala', 'Left-Accumbens-area', 'sum_Hippocampus', 'Right-Thalamus', 'Right-Caudate', 'Right-Putamen', 'Right-Pallidum', 'Right-Hippocampus', 'Right-Amygdala', 'Right-Accumbens-area', 'CSF', 'mean_inferiortemporal_thickness', 'mean_middletemporal_thickness', 'mean_temporalpole_thickness', 'mean_superiorfrontal_thickness', 'mean_superiorparietal_thickness', 'mean_supramarginal_thickness', 'mean_precuneus_thickness', 'mean_superiortemporal_thickness', 'mean_inferiorparietal_thickness', 'mean_rostralmiddlefrontal_thickness']   #  'GM_Volume', 'WM_Volume'
csf_feature = ['Amyloid_Status']


def parse_arguments():
    parser = argparse.ArgumentParser(description="Preprocess the ADNI Data...")
    parser.add_argument('--path', type=str, default="/Users/shahzadali/Documents/Datasets/ADNI", help="Directory path for clinical data files") # '/content/my_project' | "/Users/shahzadali/Documents/Datasets/ADNI"
    parser.add_argument('--file_name', type=str, default='110225_AD_data.xlsx', help="Name of the clinical data Excel file")  
    parser.add_argument('--sheet_ADNI', type=str, default='ADNI_CMSGTMMSE_A+', help="Sheet name in the Excel file")   # ADNI_CMSGT_mmse_BlncDX | ADNI_CMSGT_mmse_BlncRG
    
    parser.add_argument('--target_column', type=str, default= 'deltaMMSE_Years', choices=['deltaMMSE/Years'], help="Specify which column to use as the target")
    
    parser.add_argument('--output_dir', type=str, default='OutputDir_LCDP_NCV', help="Main project directory for output data")
    
    parser.add_argument('--scaler', type=str, default='minmax', choices=['standard', 'minmax'], help="Type of scaler to use")
    parser.add_argument('--test_size', type=float, default=0.20, help="Proportion of the dataset to include in the test split")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for data shuffling and splitting")
    parser.add_argument('--use_smote', action='store_true', help="Apply SMOTE for balancing classes in the training data")
    return parser.parse_args()



def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    args = parse_arguments()

    # General Preprocessing (Stage 1)
    file_path = os.path.join(args.path, args.file_name)
    df_raw = pd.read_excel(file_path, sheet_name=args.sheet_ADNI)

    general_preprocessing(df_raw, args.output_dir) # save data in file named '1_ADNI_generalPreprocessed.csv'

    # Load preprocessed data file
    preprocessed_file = os.path.join(args.output_dir, '1_Selected_Data.csv')

    # Feature sets dictionary for naming and processing       
    feature_combinations = {

        '1_MO': csf_feature + morphometric_features,
        '2_MS': csf_feature + microstructural_features,
        '3_GT': csf_feature + GT_global_metrics + GT_local_metrics,
        '4_MO_MS': csf_feature + microstructural_features + morphometric_features,
        '5_MO_MS_GT': csf_feature + microstructural_features + morphometric_features + GT_global_metrics + GT_local_metrics,
        '6_DG_MO_MS_GT': csf_feature + demographic_features + microstructural_features + morphometric_features + GT_global_metrics + GT_local_metrics,
        }
    # Loop through each feature combination and perform processing
    for feature_name, selected_features in feature_combinations.items():
        regression_specific_processing(
            preprocessed_file,
            target_column=args.target_column,
            feature_combination_name=feature_name,
            selected_features=selected_features,
            output_dir=args.output_dir,
            scaler_type=args.scaler,
            test_size=args.test_size,
            seed=args.seed,
            use_smote=args.use_smote
        )

if __name__ == "__main__":
    main()
