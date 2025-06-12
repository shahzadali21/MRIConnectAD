# Copyright: Shahzad Ali
# Author: Shahzad Ali
# e-mail: shahzad.ali6@unibo.it
# Created: 2024-10-24
# Last modified: 2024-10-24

"""
This script contains utility functions used across the project.
It includes functions for saving and loading models, metrics, and results,
as well as other common tasks required by the different modules.
"""

import os
import joblib
import logging
import pandas as pd


def load_data(file_path):
    clinical_df = pd.read_csv(file_path, index_col=False)
    return clinical_df

def save_data(data, file_path):
    """Saves the given DataFrame to the specified file path."""
    data.to_csv(file_path, index=False)
    logging.info(f"Data saved to {file_path}")

def save_model(model, model_name, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    file_path = os.path.join(output_dir, f"{model_name}.joblib")
    joblib.dump(model, file_path)
    logging.info(f"Model saved to {file_path}")

def load_model(model_name, input_dir):
    # Construct the full path to the model file
    file_path = os.path.join(input_dir, f"{model_name}.joblib")
    
    # Check if the file exists
    if not os.path.exists(file_path):
        logging.error(f"Model file not found: {file_path}")
        raise FileNotFoundError(f"No model found at {file_path}")
    
    # Load the model using joblib
    try:
        model = joblib.load(file_path)
        logging.info(f"Model successfully loaded from {file_path}")
        return model
    except Exception as e:
        logging.error(f"Error loading model from {file_path}: {e}")
        raise RuntimeError(f"Failed to load model {model_name} from {file_path}")

def save_results(y_test, predictions_dict, output_file):
    """
    Save the actual and predicted results to a CSV file.
    
    Parameters:
    y_test : array-like : The actual target values.
    predictions_dict : dict : Dictionary of model names and their predictions.
    output_file : str : The path where the results will be saved.
    """
    results_df = pd.DataFrame({'Actual': y_test})  # Ensure y_test is a 1D array
    for model_name, predictions in predictions_dict.items():
        results_df[f'Pred_{model_name}'] = predictions

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    results_df.to_csv(output_file, index=False)
    logging.info(f"Evaluation results/predictions saved to {output_file}")


    
def save_metrics(metrics, output_file):
    """ Save the evaluation metrics to a CSV file with a header for the model names. """
    # Ensure the index (model names) has a name
    metrics.index.name = 'Model'
    
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save the metrics DataFrame to a CSV file
    metrics.to_csv(output_file, index=True)
    
    print(f"Metrics saved to {output_file}")

def save_metrics_to_excel_v0(metrics_dict, output_file):
    """ Save all evaluation metrics to a single Excel file with multiple sheets. """
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        for classification_type, metrics in metrics_dict.items():
            metrics.to_excel(writer, sheet_name=classification_type, index_label='Model')
            logging.info(f"Metrics saved to {output_file} in sheet: {classification_type}")

def save_metrics_to_excel(metrics_dict, output_file):
    """ Save all evaluation metrics to an Excel file with multiple sheets, updating existing sheets if necessary. """
    try:
        # Check if the file exists to avoid overwriting existing data
        if os.path.exists(output_file):
            with pd.ExcelWriter(output_file, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                for classification_type, metrics in metrics_dict.items():
                    metrics.to_excel(writer, sheet_name=classification_type, index_label='Model')
                    logging.info(f"Metrics updated in {output_file} in sheet: {classification_type}")
        else:
            # If the file doesn't exist, create a new one
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                for classification_type, metrics in metrics_dict.items():
                    metrics.to_excel(writer, sheet_name=classification_type, index_label='Model')
                    logging.info(f"Metrics saved to new {output_file} in sheet: {classification_type}")
    except Exception as e:
        logging.error(f"Error saving metrics to Excel: {e}")


def save_regression_metrics(metrics_dict, output_file):
    """
    Save all evaluation metrics to an Excel file with multiple sheets, 
    updating existing sheets if necessary, without creating duplicate sheets.
    """
    try:
        # Check if the file exists
        if os.path.exists(output_file):
            # Load the existing file
            with pd.ExcelWriter(output_file, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                for subset_name, metrics in metrics_dict.items():
                    metrics.to_excel(writer, sheet_name=subset_name, index_label='Model')
                    logging.info(f"Metrics updated in {output_file} in sheet: {subset_name}")
        else:
            # Create a new file if it doesn't exist
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                for subset_name, metrics in metrics_dict.items():
                    metrics.to_excel(writer, sheet_name=subset_name, index_label='Model')
                    logging.info(f"Metrics saved to new {output_file} in sheet: {subset_name}")
    except Exception as e:
        logging.error(f"Error saving metrics to Excel: {e}")


def create_directory_structure(output_dir, classification_type, comparison):
    if classification_type == 'binary':
        folder_name = comparison.replace('vs_', '')    #f"{comparison.replace('vs_', '')}"
    else:
        folder_name = 'CN_MCI_AD'

    # Create directories for saving data, models, and results
    data_dir = os.path.join(output_dir, folder_name, 'data')
    models_dir = os.path.join(output_dir, folder_name, 'models')
    results_dir = os.path.join(output_dir, folder_name, 'results')

    # Ensure directories exist
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    return data_dir, models_dir, results_dir