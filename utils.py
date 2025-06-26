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