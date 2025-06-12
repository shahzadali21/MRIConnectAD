# Copyright: Shahzad Ali
# Author: Shahzad Ali
# e-mail: shahzad.ali6@unibo.it
# Created: 2024-10-24
# Last modified: 2024-11-12

"""
This script handles the training, optimization, and ensemble of machine learning models.
It loads preprocessed data, performs grid search optimization, evaluates the models,
calculates diversity using the Q metric, and performs ensemble voting with diverse models.
"""

# Before using this script, run `preprocessing_v2.py`.

import os
import logging
import argparse
import pandas as pd
import warnings

from utils import load_data, save_model, load_model, save_results, save_metrics_to_excel
from models import (
    get_models_and_params,
    run_nested_evaluations,
    get_ensemble_model_from_top_models,
    nested_cv_evaluation
)



import warnings
# Suppress all warnings
warnings.filterwarnings("ignore")

def create_directory_structure(output_dir, feature_combination_name, classification_type, comparison):
    feature_combination_dir = os.path.join(output_dir, feature_combination_name)
    os.makedirs(feature_combination_dir, exist_ok=True)

    comparison_folder_name = comparison.replace('vs_', '') if classification_type == 'binary' else 'CN_MCI_AD'
    classification_dir = os.path.join(feature_combination_dir, comparison_folder_name)
    os.makedirs(os.path.join(classification_dir, 'data'), exist_ok=True)
    os.makedirs(os.path.join(classification_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(classification_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(classification_dir, 'results'), exist_ok=True)

    return classification_dir


def train_and_evaluate_models(data_dir, models_dir, results_dir, classification_type, comparison, metrics_dict):
    logging.info("Loading preprocessed data")
    X_train = load_data(os.path.join(data_dir, 'X_train.csv'))
    X_test = load_data(os.path.join(data_dir, 'X_test.csv'))
    y_train = load_data(os.path.join(data_dir, 'y_train.csv')).values.ravel()
    y_test = load_data(os.path.join(data_dir, 'y_test.csv')).values.ravel()

    X_full = pd.concat([X_train, X_test], axis=0)
    y_full = pd.concat([pd.Series(y_train), pd.Series(y_test)], axis=0)

    logging.info("Fetching model definitions and parameter grids")
    models_with_params = get_models_and_params(seed=42)

    logging.info("Running Nested Cross-Validation on full dataset")
    nested_results, nested_summary = run_nested_evaluations(models_with_params, X_full, y_full)
    nested_summary.to_csv(os.path.join(results_dir, "nested_cv_summary.csv"), index=False)
    logging.info(f"Nested CV summary saved to: {results_dir}")

    metrics_dict[f"{classification_type}_{comparison}"] = nested_summary

    logging.info("Building ensemble model using top models")
    ensemble_model = get_ensemble_model_from_top_models(
        summary_df=nested_summary,
        models_dict=models_with_params,
        strategy='topk',  # or 'mean'
        top_k=3,
        seed=42
    )

    logging.info("Training ensemble model on training set")
    ensemble_model.fit(X_train, y_train)

    logging.info("Evaluating ensemble model")
    predictions = {}
    metrics, predictions = nested_cv_evaluation({"Ensemble": ensemble_model}, X_test, y_test)

    logging.info("Saving metrics for all models")
    metrics_dict[f"{classification_type}_{comparison}"] = metrics
    print("Model Performance Evaluation Metrics:")
    print(metrics.sort_values(by='Accuracy', ascending=False))

    save_results(y_test, predictions, os.path.join(results_dir, 'predictions.csv'))
    

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train and evaluate models for all classification types")
    parser.add_argument('--output_dir', type=str, default='DSC_NCV', help="Main project directory for clinical AD dataset")
    parser.add_argument('--n_jobs', type=int, default=-1, help="Number of parallel jobs to run (-1 for all cores)")
    return parser.parse_args()

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    args = parse_arguments()

    # Dictionary to hold metrics for each classification type
    metrics_dict = {}

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
        for classification_type, comparisons in CLASSIFICATION_COMPARISONS.items():
            for comparison in comparisons:
                logging.info("#" * 60)
                logging.info(f"Processing {classification_type} classification: {comparison}")
                logging.info("#" * 60)

                classification_dir = create_directory_structure(
                    output_dir=args.output_dir,
                    feature_combination_name=feature_combination_name,
                    classification_type=classification_type,
                    comparison=comparison
                )

                data_dir = os.path.join(classification_dir, 'data')
                models_dir = os.path.join(classification_dir, 'models')
                results_dir = os.path.join(classification_dir, 'results')

                train_and_evaluate_models(data_dir, models_dir, results_dir, classification_type, comparison, metrics_dict)

        output_file = os.path.join(args.output_dir, f'ClassificationMetrics_{feature_combination_name}.xlsx')
        save_metrics_to_excel(metrics_dict, output_file)

if __name__ == "__main__":
    main()
