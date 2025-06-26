# module: DSC_explainability.py
# -*- coding: utf-8 -*-
# Author: Shahzad Ali

"""
This module generates various plots for model comparison as well as explainability using SHAP and LIME.
It loads trained models and test data, generates SHAP and LIME explanations,
and saves the resulting visualizations and explanations to the specified output directory.
"""

# Before using this script, run `preprocessing.py` and `model_training.py`.


import os
import argparse
import logging
import pandas as pd
from pathlib import Path

from DSC_model_utils import load_model, load_data
import DSC_XAI_utils as xai

import warnings
warnings.filterwarnings('ignore')

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


def get_class_mapping(comparison):
    if comparison == 'CN_vs_AD':
        return {0: 'CN', 1: 'AD-D'}
    elif comparison == 'CN_vs_MCI':
        return {0: 'CN', 1: 'MCI'}
    elif comparison == 'MCI_vs_AD':
        return {0: 'MCI', 1: 'AD-D'}
    elif comparison == 'CN_MCI_AD':
        return {0: 'CN', 1: 'MCI', 2: 'AD-D'}
    return {}


def generate_comparison_plots(data_dir, models_dir, results_dir, plots_dir, output_dir, metrics_file, classification_type, comparison):
    # Load the evaluation metrics from the specified Excel file
    logging.info(f"Loading evaluation metrics from {metrics_file}")

    # The sheet name corresponds to "{classification_type}_{comparison}"
    sheet_name = f"{classification_type}_{comparison}"
    eval_metrics = pd.read_excel(metrics_file, sheet_name=sheet_name)
    print("Evaluation Metrics:\n", eval_metrics)

    # Extract the model names and load the trained models
    logging.info("Loading trained models")
    model_names = eval_metrics['Model'].tolist()
    print(model_names)

    # Load trained Decision Tree model
    model_name = 'DT'
    logging.info(f"Loading the trained Decision Tree '{model_name}' model")
    #model = trained_models['DT']
    model = load_model(model_name=model_name, input_dir=models_dir)
    feature_names = pd.read_csv(os.path.join(data_dir, 'X_train.csv')).columns

    logging.info(f"Generating the decision tree plot using '{model_name}'")
    xai.plot_decision_tree(model=model, feature_names=feature_names, output_dir=plots_dir)

    # Load trained Random Forest model
    rf_model_name = 'RF'
    logging.info(f"Loading the trained '{rf_model_name}' model")
    rf_model = load_model(model_name=rf_model_name, input_dir=models_dir)
    logging.info(f"Plotting feature importance for Random Forest '{rf_model_name}'")
    xai.plot_feature_importance(model=rf_model, feature_names=feature_names, top_n=5, results_dir=results_dir, output_dir=plots_dir)
 

    logging.info("Generating model performance comparison plot")
    xai.plot_model_performance_comparison(metrics=eval_metrics, model_names=model_names, output_dir=plots_dir, filename_prefix="Comparison_ModelEvaluationMetrics")

    logging.info("Generating model accuracy plot of all models")
    xai.plot_model_accuracy(metrics=eval_metrics, model_names=model_names, output_dir=plots_dir, filename_prefix="Comparison_ModelAccuracy")



def generate_shap_explainability(data_dir, models_dir, results_dir, plots_dir, class_mapping):
    # Load the trained Decision Tree model
    model_name = 'best_LR-SGD'
    logging.info(f"Loading the trained Decision Tree '{model_name}' model")
    #model = load_model(model_name=model_name, input_dir=models_dir)
    #models_dir = Path(models_dir)
    model_path = models_dir / f"{model_name}.pkl"
    model = load_model(model_path)


    # Load the training and test data
    X_train = load_data(os.path.join(data_dir, 'X_train.csv'))
    X_test = load_data(os.path.join(data_dir, 'X_test.csv'))
    y_test = load_data(os.path.join(data_dir, 'y_test.csv')).values.ravel()

    logging.info("############ Explainability using SHAP ############")

    logging.info(f"Plotting SHAP summary for each class using {model_name}")
    xai.plot_shap_summary_by_class(model=model, X_test=X_test, class_mapping=class_mapping, output_dir=plots_dir)

    logging.info(f"Plotting aggregated SHAP summary using {model_name}")
    xai.plot_shap_aggregated_summary(model=model, X_test=X_test, output_dir=plots_dir)

    logging.info("Plotting SHAP dependence and feature importance for all features")
    xai.plot_shap_dependence_and_feature_importance(model=model, X_test=X_test, class_mapping=class_mapping, output_dir=plots_dir, results_dir=results_dir)

    logging.info(f"Plotting SHAP decision plots for all samples for each class using {model_name}")
    xai.plot_decision_for_all_samples_by_class(model=model, X_test=X_test, y_test=y_test, class_mapping=class_mapping, output_dir=plots_dir)

    logging.info(f"Plotting SHAP Aggregated Waterfall plots for all samples for each class using {model_name}")
    xai.plot_waterfall_aggregated_by_class(model=model, X_test=X_test, y_test=y_test, class_mapping=class_mapping, output_dir=plots_dir)
    xai.plot_waterfall_for_all_classes_combined(model=model, X_test=X_test, y_test=y_test, class_mapping=class_mapping, output_dir=plots_dir)

    logging.info(f"Plotting SHAP Aggregated Waterfall plots for all samples for each class using {model_name}")
    xai.plot_beeswarm_for_all_classes(model=model, X_test=X_test, y_test=y_test, class_mapping=class_mapping, output_dir=plots_dir)

    logging.info(f"Plotting SHAP Barplot using {model_name}")
    xai.plot_shap_bar_for_all_classes(model=model, X_test=X_test, y_test=y_test, class_mapping=class_mapping, output_dir=plots_dir)
    xai.plot_shap_bar_multiclass_sorted(model=model, X_test=X_test, y_test=y_test, class_mapping=class_mapping, output_dir=plots_dir)


    logging.info("############ Explainability using LIME ############")
    logging.info(f"Generating LIME explanation for a specific sample using {model_name}")
    xai.plot_lime_explanation(model=model, X_train=X_train, X_test=X_test, class_mapping=class_mapping, index=29, num_features=5, output_dir=plots_dir)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate visualizations and explainability for models")
    parser.add_argument('--output_dir', type=str, default='DSC_NCV', help="Main project directory for clinical AD dataset")  #V6_ProjectOutput_AmyStatus_Ens_Top2 | V6_ProjectOutput_AmyStatus_Ens_aboveMean
    return parser.parse_args()

def main():
    # Configure logging to display INFO level messages
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    args = parse_arguments()


    # Loop over all feature combination folders
    feature_combinations = ['MO',]
                            #'MS', 'GT', 'MO_MS', 'MO_MS_GT', 'DG_MO_MS_GT']

    # Define classification types and comparisons
    CLASSIFICATION_COMPARISONS = {
            'binary': ['CN_vs_AD', 'CN_vs_MCI', 'MCI_vs_AD'],
            'three_level': ['CN_MCI_AD']
        }
    
    for feature_combination_name in feature_combinations:
        # Set the metrics file path in the main directory
        metrics_file = os.path.join(args.output_dir, f'ClassificationMetrics_{feature_combination_name}.xlsx')     
        
        # Loop over all classification types and comparisons
        for classification_type, comparisons in CLASSIFICATION_COMPARISONS.items():
            for comparison in comparisons:
                logging.info(f"###############################################################")
                logging.info(f"Processing {classification_type} classification: {comparison}")
                logging.info(f"###############################################################")

                classification_dir = create_directory_structure(
                    output_dir=args.output_dir,
                    feature_combination_name=feature_combination_name,
                    classification_type=classification_type,
                    comparison=comparison
                )
                data_dir = os.path.join(classification_dir, 'data')
                models_dir = os.path.join(classification_dir, 'models')
                plots_dir = os.path.join(classification_dir, 'plots')
                results_dir = os.path.join(classification_dir, 'results')

                # Get the correct class mapping based on the comparison
                class_mapping = get_class_mapping(comparison)

                # Generate model comparison plots
                #generate_comparison_plots(data_dir, models_dir, results_dir, plots_dir, args.output_dir, metrics_file, classification_type, comparison)
                
                # Generate model explainability plots 
                generate_shap_explainability(
                                    data_dir=Path(data_dir), 
                                    models_dir=Path(models_dir),
                                    results_dir=Path(results_dir),
                                    plots_dir=Path(plots_dir),
                                    class_mapping=class_mapping
                                )


if __name__ == "__main__":
    main()
