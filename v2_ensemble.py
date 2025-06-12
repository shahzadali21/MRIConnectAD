# Copyright: Shahzad Ali
# Author: Shahzad Ali
# e-mail: shahzad.ali6@unibo.it
# Created: 2024-10-24
# Last modified: 2024-11-12

"""
This script handles the selection, evaluation, and ensemble of machine learning models.
It reads performance metrics from an Excel file, selects models based on specified criteria,
and creates ensemble classifiers. The results are appended to an existing Excel file in the main project directory.
"""
# Before using this script, run `preprocessing_v2.py` and `model_optimizing_v2.py`.

import os
import joblib
import logging
import argparse
import pandas as pd
from sklearn.ensemble import VotingClassifier

from utils import load_data, save_results, save_model, save_metrics_to_excel
from models import evaluate_models

import warnings
# Suppress all warnings
warnings.filterwarnings("ignore")


# Create directory structure for feature combinations and comparisons
def create_directory_structure(output_dir, feature_combination_name, classification_type, comparison):
    feature_combination_dir = os.path.join(output_dir, feature_combination_name)
    os.makedirs(feature_combination_dir, exist_ok=True)

    # Create subdirectory for each classification comparison within the feature combination directory
    comparison_folder_name = comparison.replace('vs_', '') if classification_type == 'binary' else 'CN_MCI_AD'
    classification_dir = os.path.join(feature_combination_dir, comparison_folder_name)
    os.makedirs(os.path.join(classification_dir, 'data'), exist_ok=True)
    os.makedirs(os.path.join(classification_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(classification_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(classification_dir, 'results'), exist_ok=True)

    return classification_dir

# Load metrics from Excel file
def load_metrics_from_excel(metrics_file_path, sheet_name):
    try:
        # Load the specified sheet from the Excel file
        metrics_df = pd.read_excel(metrics_file_path, sheet_name=sheet_name, index_col=0)
        return metrics_df
    except Exception as e:
        logging.error(f"Error loading metrics from Excel: {e}")
        return None


def save_selected_models(output_dir, feature_combination_name, comparison, selected_model_names):
    """Saves the selected model names for ensemble learning to a CSV file."""
    # Define the path for the selected models CSV file
    selected_models_file = os.path.join(output_dir, 'selected_models.csv')
    
    # Check if the file exists
    if os.path.exists(selected_models_file):
        # If the file exists, append to it
        selected_models_df = pd.read_csv(selected_models_file)
    else:
        # If the file doesn't exist, create a new DataFrame with headers
        selected_models_df = pd.DataFrame(columns=['Feature_Combination', 'Classification', 'Selected_Models'])
    
    # Convert selected_model_names list to a string (comma-separated)
    selected_models_str = ', '.join(selected_model_names)
    
    # Create a new row to append to the DataFrame
    new_row = pd.DataFrame({'Feature_Combination': [feature_combination_name],
                            'Classification': [comparison],
                            'Selected_Models': [selected_models_str]})
    selected_models_df = pd.concat([selected_models_df, new_row], ignore_index=True)
    
    # Save the updated DataFrame back to the CSV file
    selected_models_df.to_csv(selected_models_file, index=False)


# Ensemble voting function
def ensemble_voting(output_dir, results_dir, feature_combination_name, classification_type, comparison, metrics_dict):
    # Load the preprocessed data
    data_dir = os.path.join(output_dir, feature_combination_name, comparison.replace('vs_', '') if classification_type == 'binary' else 'CN_MCI_AD', 'data')
    models_dir = os.path.join(output_dir, feature_combination_name, comparison.replace('vs_', '') if classification_type == 'binary' else 'CN_MCI_AD', 'models')
    metrics_file_path = os.path.join(output_dir, f'ClassificationMetrics_{feature_combination_name}.xlsx')
    metrics_df = load_metrics_from_excel(metrics_file_path, sheet_name=f'{classification_type}_{comparison}')

    if metrics_df is None:
        logging.error(f"No metrics found for {classification_type} classification {comparison}. Skipping ensemble creation.")
        return

    # Load training and test data
    X_train = load_data(os.path.join(data_dir, 'X_train.csv'))
    X_test = load_data(os.path.join(data_dir, 'X_test.csv'))
    y_train = load_data(os.path.join(data_dir, 'y_train.csv')).values.ravel()
    y_test = load_data(os.path.join(data_dir, 'y_test.csv')).values.ravel()

    # Load optimized models
    #optimized_models = {model_name: joblib.load(os.path.join(models_dir, f'{model_name}.joblib')) for model_name in metrics_df.index}
    optimized_models = {}
    base_classifiers = [name for name in metrics_df.index if name not in ['Ensemble_hV', 'Ensemble_sV']]  # Exclude ensemble classifiers

    for model_name in base_classifiers:
        model_path = os.path.join(models_dir, f'{model_name}.joblib')
        if os.path.exists(model_path):
            optimized_models[model_name] = joblib.load(model_path)
        else:
            logging.warning(f"Model file {model_name}.joblib not found. Skipping this model.")

    # # Step 4: Select models based on the "above mean accuracy" method and exclude ensemble classifiers
    # mean_accuracy = metrics_df['Accuracy'].mean()
    # logging.info(f"Mean model accuracy: {mean_accuracy}")

    # excluded_models = ['Ensemble_hV', 'Ensemble_sV']  # List of models to exclude

    # # Select models based on the "ABOVE MEAN ACCURACY" method
    # selected_model_names = [model for model in metrics_df[metrics_df['Accuracy'] > mean_accuracy].index if model not in excluded_models]
    # logging.info(f"Selected models with accuracy above mean: {selected_model_names}")

    # # Select TOP 3 or 5 Models based on accuracy
    # # Sort models by accuracy in descending order and exclude specified models
    # #sorted_models = metrics_df.sort_values(by='Accuracy', ascending=False).index
    # #filtered_models = [model for model in sorted_models if model not in excluded_models]
    # #top_n = 5
    # #selected_model_names = filtered_models[:top_n]
    # #logging.info(f"Selected top 5 models for the ensemble: {selected_model_names}")

    # if not selected_model_names:
    #     logging.error("No models were selected for the ensemble. Skipping.")
    #     return

    # Step 4: Select models based on either "above mean accuracy" or "top N models" method
    mean_accuracy = metrics_df['Accuracy'].mean()
    logging.info(f"Mean model accuracy: {mean_accuracy}")

    excluded_models = ['Ensemble_hV', 'Ensemble_sV']  # List of models to exclude

    # Selection method: Choose between 'above_mean' or 'top_n'
    selection_method = 'top_n'  # 'above_mean' | 'top_n'

    if selection_method == 'above_mean':
        # Select models with accuracy above mean, excluding specified models
        selected_model_names = [
            model for model in metrics_df[metrics_df['Accuracy'] > mean_accuracy].index 
            if model not in excluded_models
        ]
        logging.info(f"Selected models with accuracy above mean: {selected_model_names}")

    elif selection_method == 'top_n':
        # Select the top N models based on accuracy, excluding specified models
        top_n = 5   # Number of TOP MODELS to select
        sorted_models = metrics_df.sort_values(by='Accuracy', ascending=False).index
        filtered_models = [model for model in sorted_models if model not in excluded_models]
        selected_model_names = filtered_models[:top_n]
        logging.info(f"Selected top {top_n} models for the ensemble: {selected_model_names}")

    if not selected_model_names:
        logging.error("No models were selected for the ensemble. Skipping.")
        return


    # Save the selected base models to the CSV file
    save_selected_models(output_dir, feature_combination_name, comparison, selected_model_names)

    selected_models = [(name, optimized_models[name]) for name in selected_model_names]
    
    # Step 5: Ensemble voting with selected models
    logging.info(f"Creating ensemble with selected models using 'above mean accuracy' method")
    ensemble_hardVoting = VotingClassifier(estimators=selected_models, voting='hard')   # Hard Voting Classifier
    ensemble_softVoting = VotingClassifier(estimators=selected_models, voting='soft')   # Soft Voting Classifier

    # Fit the ensemble models
    ensemble_hardVoting.fit(X_train, y_train)
    ensemble_softVoting.fit(X_train, y_train)
    
    # Make predictions and evaluate
    y_pred_ensemble_hardVoting = ensemble_hardVoting.predict(X_test)
    y_pred_ensemble_softVoting = ensemble_softVoting.predict(X_test)

    # Evaluate Voting Classifiers
    logging.info("Evaluating Voting Classifiers")
    hard_voting_metrics, _ = evaluate_models(models={'Ensemble_hV': ensemble_hardVoting}, X_test=X_test, y_test=y_test)
    soft_voting_metrics, _ = evaluate_models(models={'Ensemble_sV': ensemble_softVoting}, X_test=X_test, y_test=y_test)

    # Debug: Print or log the metrics structure
    logging.info(f"Hard voting metrics: {hard_voting_metrics}")
    logging.info(f"Soft voting metrics: {soft_voting_metrics}")

    # Save Ensemble/Voting Classifiers
    logging.info("Saving Voting Classifiers")
    save_model(ensemble_hardVoting, 'Ensemble_hV', models_dir)
    save_model(ensemble_softVoting, 'Ensemble_sV', models_dir)


    # Combine metrics from voting classifiers with other models and save
    logging.info("Saving combined metrics for all models")
    combined_metrics = pd.concat([metrics_df, hard_voting_metrics, soft_voting_metrics])
    metrics_dict[f"{classification_type}_{comparison}"] = combined_metrics

    # Save predictions to CSV in results directory
    combined_predictions = {model_name: model.predict(X_test) for model_name, model in optimized_models.items()}
    combined_predictions['Ensemble_hV'] = y_pred_ensemble_hardVoting
    combined_predictions['Ensemble_sV'] = y_pred_ensemble_softVoting
    
    # Combine all predictions, including those from the ensemble models
    #combined_predictions = {**predictions, 'Ensemble_hV': y_pred_ensemble_hardVoting, 'Ensemble_sV': y_pred_ensemble_softVoting}
    
    logging.info("Saving predictions")
    save_results(y_test, combined_predictions, os.path.join(results_dir, 'predictions.csv'))

    print("Model Performance Evaluation Metrics:")
    print(combined_metrics.sort_values(by='Accuracy', ascending=False))


def parse_arguments():
    parser = argparse.ArgumentParser(description="Create an ensemble using Voting Classifier for all classification types and comparisons")
    parser.add_argument('--output_dir', type=str, default='V6_ProjectOutput_AmyStatus_Ens_Top5', help="Main project directory for clinical AD dataset")
    return parser.parse_args()

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    args = parse_arguments()
    
    # Dictionary to hold metrics for each classification type
    metrics_dict = {}

    # Loop over all feature combination folders
    #feature_combination_folders = [folder for folder in os.listdir(args.output_dir) if os.path.isdir(os.path.join(args.output_dir, folder))]
    # OR 
    # Loop over specified feature combination folders
    feature_combinations = [
                            'MO',
                            'MS',
                            'MO_MS_GT',
                            'DG_MO_MS_GT'
                            ]
    
    # Define classification types and comparisons
    CLASSIFICATION_COMPARISONS = {
        'binary': ['CN_vs_AD', 'CN_vs_MCI', 'MCI_vs_AD'],
        'three_level': ['CN_MCI_AD']
    }

    #for feature_combination_name in feature_combination_folders:
    for feature_combination_name in feature_combinations:
        # Loop over all classification types and comparisons
        for classification_type, comparisons in CLASSIFICATION_COMPARISONS.items():
            for comparison in comparisons:
                logging.info(f"###############################################################")
                logging.info(f"Processing {classification_type} classification: {comparison}")
                logging.info(f"###############################################################")

                # Create directory structure for the current classification and comparison
                # Create directory structure for the current classification and comparison
                classification_dir = create_directory_structure(
                    output_dir=args.output_dir,
                    feature_combination_name=feature_combination_name,
                    classification_type=classification_type,
                    comparison=comparison
                )
                results_dir = os.path.join(classification_dir, 'results')

                ensemble_voting(
                    output_dir=args.output_dir,
                    results_dir=results_dir,
                    feature_combination_name=feature_combination_name,
                    classification_type=classification_type,
                    comparison=comparison,
                    metrics_dict=metrics_dict,
                )
    
        # Save updated metrics to Excel
        output_metrics_file = os.path.join(args.output_dir, f'ClassificationMetrics_{feature_combination_name}.xlsx')
        save_metrics_to_excel(metrics_dict, output_metrics_file)

if __name__ == "__main__":
    main()
