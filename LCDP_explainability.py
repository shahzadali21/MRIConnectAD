# Copyright: Shahzad Ali
# Author: Shahzad Ali
# e-mail: shahzad.ali6@unibo.it
# Created: 2025-03-21
# Last modified: 2025-04-14

"""
This module generates various plots for model comparison as well as explainability using SHAP and LIME.
It loads trained models and test data, generates SHAP and LIME explanations,
and saves the resulting visualizations and explanations to the specified output directory.
"""

# Before using this script, run `preprocessing_v2.py` and `model_optimizing_v2.py` and `Voting_2.py`.


import os
import argparse
import logging
import pandas as pd

from utils import load_model, load_data

from LCDP_model_utils import select_top_models

import LCDP_visualization as viz
import LCDP_XAI_reression as xai


import warnings
warnings.filterwarnings('ignore')

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

def generate_comparison_plots(data_dir, models_dir, results_dir, plots_dir, output_dir, metrics_file, subset_name):
    # Load the evaluation metrics from the specified Excel file
    logging.info(f"Loading evaluation metrics from {metrics_file}")
    sheet_name = subset_name
    eval_metrics = pd.read_excel(metrics_file, sheet_name=sheet_name)
    print("Evaluation Metrics:\n", eval_metrics)

    # Extract the model names and load the trained models
    logging.info("Loading trained models")
    model_names = eval_metrics['Model'].tolist()
    print(model_names)
    all_models = {name: load_model(name, models_dir) for name in model_names}

    # Select top models based on R² for visualization
    logging.info(f"Selecting top models for regression comparison")
    top_models = select_top_models(eval_metrics, all_models, top_n=4)
    logging.info(f"Top models selected for regression comparison: {[name for name, _ in top_models]}")

    # Load test data
    X_test = load_data(os.path.join(data_dir, 'X_test.csv'))
    y_test = load_data(os.path.join(data_dir, 'y_test.csv')).values.ravel()

    # Generate comparison plots
    logging.info("Generating model performance comparison plot")
    viz.plot_model_performance_comparison(metrics=eval_metrics, model_names=model_names, output_dir=plots_dir, filename_prefix="Regression_ModelEvaluationMetrics")

    logging.info("Generating RMSE plot of all models")
    viz.plot_rmse(metrics=eval_metrics, model_names=model_names, output_dir=plots_dir, filename_prefix="Regression_ModelRMSE")

    logging.info("Generating R² plot for all models")
    viz.plot_r2(metrics=eval_metrics, model_names=model_names, output_dir=plots_dir)

    logging.info("Regression comparison and plot generation complete.")

    # Load the trained Random Forest model
    rf_model_name = 'RF'
    logging.info(f"Loading the trained '{rf_model_name}' model")
    rf_model = load_model(model_name=rf_model_name, input_dir=models_dir)
    feature_names = pd.read_csv(os.path.join(data_dir, 'X_train.csv')).columns

    logging.info(f"Plotting feature importance for Random Forest '{rf_model_name}'")
    viz.plot_feature_importance(model=rf_model, feature_names=feature_names, top_n=5, results_dir=results_dir, output_dir=plots_dir)
 



def generate_shap_explainability(data_dir, models_dir, plots_dir, results_dir):
    # Load the trained model with the best MAE
    best_model_name = 'AdaB' # 'SVR' | 'AdaB'
    logging.info(f"Loading the trained regression model '{best_model_name}'")
    best_model = load_model(model_name=best_model_name, input_dir=models_dir)

    
    # Load the training and test data
    X_train = load_data(os.path.join(data_dir, 'X_train.csv'))
    X_test = load_data(os.path.join(data_dir, 'X_test.csv'))
    feature_names = pd.read_csv(os.path.join(data_dir, 'X_train.csv')).columns

    logging.info("############ Explainability using SHAP for Regression ############")

    logging.info(f"Plotting SHAP summary for {best_model_name}")
    xai.plot_shap_summary_regression(model=best_model, X_test=X_test, output_dir=plots_dir)

    logging.info(f"Plotting SHAP feature importance for {best_model_name}")
    xai.plot_shap_feature_importance(model=best_model, X_test=X_test, output_dir=plots_dir, results_dir=results_dir)
    
    logging.info(f"Plotting SHAP Waterfall plot for {best_model_name}")
    xai.plot_shap_waterfall_regression(model=best_model, X_test=X_test, output_dir=plots_dir)

    logging.info("SHAP explainability generation complete.")

    # Generate LIME explanation for sample index 0
    xai.plot_lime_explanation_regression(model=best_model, X_train=X_train, X_test=X_test, sample_index=0, num_features=20, output_dir=plots_dir)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate visualizations and explainability for models")
    parser.add_argument('--output_dir', type=str, default='Results/LCDP', help="Main project directory for clinical AD dataset")  #V3_ProjectOutput_MMSE_Ens_Top5
    return parser.parse_args()

def main():
    # Configure logging to display INFO level messages
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    args = parse_arguments()

    # Loop over all feature combination folders
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
        # Set the metrics file path in the main directory
        metrics_file = os.path.join(args.output_dir, f'RegressionMetrics_{feature_combination_name}.xlsx')

        for subset_name in SUBSETS:
            logging.info(f"###############################################################")
            logging.info(f"Processing subset: {subset_name}")
            logging.info(f"###############################################################")

            subset_dir = create_directory_structure(
                output_dir=args.output_dir,
                feature_combination_name=feature_combination_name,
                subset_name=subset_name
            )
            data_dir = os.path.join(subset_dir, 'data')
            models_dir = os.path.join(subset_dir, 'models')
            plots_dir = os.path.join(subset_dir, 'plots')
            results_dir = os.path.join(subset_dir, 'results')

            # Generate comparison plots and SHAP explainability
            #generate_comparison_plots(data_dir, models_dir, results_dir, plots_dir, args.output_dir, metrics_file, subset_name)
            generate_shap_explainability(data_dir, models_dir, plots_dir, results_dir)


if __name__ == "__main__":
    main()