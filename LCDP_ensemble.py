# regression_ensemble.py

# Copyright: Shahzad Ali
# Author: Shahzad Ali
# e-mail: shahzad.ali6@unibo.it
# Created: 2025-03-21
# Last modified: 2025-04-14

"""
This script handles the selection, evaluation, and ensemble of regression models.
It reads performance metrics from an Excel file, selects models based on specified criteria,
and creates ensemble regressors. The results are appended to an existing Excel file in the main project directory.
"""

import os
import joblib
import logging
import argparse
import pandas as pd
from sklearn.ensemble import VotingRegressor

from LCDP_model_utils import load_data, save_results, save_model, save_metrics_to_excel, evaluate_models

import warnings
warnings.filterwarnings("ignore")


# Create directory structure for feature combinations and subsets
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

# Load metrics from Excel file
def load_metrics_from_excel(metrics_file_path, sheet_name):
    try:
        # Load the specified sheet from the Excel file
        metrics_df = pd.read_excel(metrics_file_path, sheet_name=sheet_name, index_col=0)
        return metrics_df
    except Exception as e:
        logging.error(f"Error loading metrics from Excel: {e}")
        return None
    

# Function to save selected models for the ensemble regressor
import pandas as pd

def save_selected_models_for_ensemble(output_dir, feature_combination_name, subset_name, selected_model_names):
    """Saves the selected base models for the ensemble regressor to a CSV file."""
    selected_models_file = os.path.join(output_dir, 'selected_models_for_ensemble.csv')
    
    # Check if the file exists
    if os.path.exists(selected_models_file):
        # If the file exists, read it into a DataFrame
        selected_models_df = pd.read_csv(selected_models_file)
    else:
        # If the file doesn't exist, create a new DataFrame with headers
        selected_models_df = pd.DataFrame(columns=['Feature_Combination', 'Subset', 'Selected_Models'])
    
    # Convert the selected model names list to a string (comma-separated)
    selected_models_str = ', '.join(selected_model_names)
    
    # Create a new row to append to the DataFrame
    new_row = pd.DataFrame({'Feature_Combination': [feature_combination_name],
                            'Subset': [subset_name],
                            'Selected_Models': [selected_models_str]})
    
    selected_models_df = pd.concat([selected_models_df, new_row], ignore_index=True)
    
    # Save the updated DataFrame back to the CSV file
    selected_models_df.to_csv(selected_models_file, index=False)



# Ensemble regression function
def ensemble_regression(output_dir, results_dir, feature_combination_name, subset_name, metrics_dict):
    # Load the preprocessed data
    data_dir = os.path.join(output_dir, feature_combination_name, subset_name, 'data')
    models_dir = os.path.join(output_dir, feature_combination_name, subset_name, 'models')
    metrics_file_path = os.path.join(output_dir, f'RegressionMetrics_{feature_combination_name}.xlsx')
    metrics_df = load_metrics_from_excel(metrics_file_path, sheet_name=f'{subset_name}')

    if metrics_df is None:
        logging.error(f"No metrics found for subset: {subset_name}. Skipping ensemble creation.")
        return

    # Load training and test data
    X_train = load_data(os.path.join(data_dir, 'X_train.csv'))
    X_test = load_data(os.path.join(data_dir, 'X_test.csv'))
    y_train = load_data(os.path.join(data_dir, 'y_train.csv')).values.ravel()
    y_test = load_data(os.path.join(data_dir, 'y_test.csv')).values.ravel()

    # Load optimized models
    optimized_models = {}
    base_regressors = [name for name in metrics_df.index if name not in ['Ensemble']]  # Exclude ensemble regressors

    for model_name in base_regressors:
        model_path = os.path.join(models_dir, f'{model_name}.joblib')
        if os.path.exists(model_path):
            optimized_models[model_name] = joblib.load(model_path)
        else:
            logging.warning(f"Model file {model_name}.joblib not found. Skipping this model.")


    # Step 4: Select models based on the "below_mean" or "top_n" method; 
    # Define the metric to use ('MAE', 'MSE', or 'RMSE')
    error_metric = 'MAE'  # Change this to 'MSE' or 'RMSE' as needed

    # Selection method: Choose between "below_mean" or "top_n"
    selection_method = "below_mean"  # top_n | below_mean

    if selection_method == "below_mean":
        # Step 4: Select models based on the "BELOW MEAN" method
        mean_error = metrics_df[error_metric].mean()
        logging.info(f"Mean model {error_metric}: {mean_error}")

        #selected_model_names = [model for model in metrics_df[metrics_df[error_metric] < mean_error].index]
        selected_model_names = metrics_df[metrics_df[error_metric] < mean_error].index.tolist()
        logging.info(f"Selected models with {error_metric} below mean: {selected_model_names}")

    elif selection_method == "top_n":
        top_n = 3  # Change to 5 if needed
        selected_model_names = metrics_df.nsmallest(top_n, error_metric).index.tolist()
        logging.info(f"Selected top {top_n} models with the lowest {error_metric}: {selected_model_names}")

    else:
        logging.error("Invalid selection method. Choose either 'below_mean' or 'top_n'.")
        selected_model_names = []

    # Handle case when no models are selected
    if not selected_model_names:
        logging.error("No models were selected. Skipping further steps.")
        return

    # Save the selected base models to a CSV file for future reference
    save_selected_models_for_ensemble(output_dir, feature_combination_name, subset_name, selected_model_names)

    selected_models = [(name, optimized_models[name]) for name in selected_model_names]

    # Step 5: Ensemble regression with selected models
    logging.info(f"Creating ensemble with selected models using 'above mean R²' method")
    ensemble = VotingRegressor(estimators=selected_models)
    # Fit the ensemble model
    ensemble.fit(X_train, y_train)
    save_model(ensemble, 'Ensemble', models_dir)

    # Evaluate Voting Regressor using evaluate_models
    voting_models = {'Ensemble': ensemble}
    ensemble_metrics_df, predictions = evaluate_models(voting_models, X_test, y_test)

    # Log the ensemble metrics
    logging.info(f"Ensemble metrics:\n{ensemble_metrics_df}")

    # Merge ensemble metrics with the original metrics
    metrics_dict[subset_name] = pd.concat([metrics_df, ensemble_metrics_df])
    # Save metrics and predictions
    metrics_dict[f"{subset_name}_Ensemble"] = ensemble_metrics_df

    save_results(y_test, predictions, os.path.join(results_dir, 'predictions_ensemble.csv'))


def parse_arguments():
    parser = argparse.ArgumentParser(description="Create an ensemble using Voting Regressor for all subsets")
    parser.add_argument('--output_dir', type=str, default='Results/LCDP', help="Main project directory for clinical AD dataset")
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    args = parse_arguments()

    # Dictionary to hold metrics for each subset
    metrics_dict = {}

    # Loop over specified feature combination folders
    feature_combinations = [
                            #'Clinical',
                            'MO',
                            # 'MS',
                            # 'GT_Local',
                            # 'GT_Global',
                            # 'GT',
                            # 'MO_MS',
                            # 'MO_GT',
                            # 'MS_GT',
                            # 'MO_MS_GT',
                            # 'MO_MS_GT_Dg',
                            # 'MO_MS_GT_Age',
                            # 'MO_MS_GT_Sex',
                            # 'MO_MS_GT_Edu',
                            # 'MO_MS_GT_Age_Sex',
                            ]

    # Subsets for regression
    SUBSETS = ["without_nan"]   #"without_nan_zero"

    for feature_combination_name in feature_combinations:
        # Loop over regression subsets
        for subset_name in SUBSETS:
            logging.info(f"###############################################################")
            logging.info(f"Processing subset: {subset_name}")
            logging.info(f"###############################################################")

            # Create directory structure for the current subset
            subset_dir = create_directory_structure(output_dir=args.output_dir, feature_combination_name=feature_combination_name, subset_name=subset_name)
            results_dir = os.path.join(subset_dir, 'results')

            # Perform ensemble regression for the current subset
            ensemble_regression(args.output_dir, results_dir, feature_combination_name, subset_name, metrics_dict)

        # Save updated metrics to Excel
        output_metrics_file = os.path.join(args.output_dir, f'RegressionMetrics_{feature_combination_name}.xlsx')
        save_metrics_to_excel(metrics_dict, output_metrics_file)


if __name__ == "__main__":
    main()
