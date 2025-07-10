# module: LCDP_model_training.py
# -*- coding: utf-8 -*-
# Author: Shahzad Ali
"""
Train, evaluate, and ensemble regression models using nested CV and bootstrapping.
"""

import os
import logging
import argparse
import numpy as np
import pandas as pd

from LCDP_model_utils import (
    get_models_and_params,
    nested_cv_regression,
    build_voting_regressor,
    evaluate_regression_ensemble_bootstrap,
    save_model, load_model,
    evaluate_models,
    load_data, save_data,
    save_results, save_metrics_to_excel
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def create_directory_structure(output_dir, feature_combo, subset_name):
    subset_dir = os.path.join(output_dir, feature_combo, subset_name)
    for sub in ['data', 'models', 'results', 'plots']:
        os.makedirs(os.path.join(subset_dir, sub), exist_ok=True)
    return subset_dir


def train_and_evaluate_models(data_dir, models_dir, results_dir, subset_name, metrics_dict):
    logging.info(f"Loading data for subset: {subset_name}")
    X_train = load_data(os.path.join(data_dir, 'X_train.csv')).to_numpy()
    X_test = load_data(os.path.join(data_dir, 'X_test.csv')).to_numpy()
    y_train = load_data(os.path.join(data_dir, 'y_train.csv')).squeeze().to_numpy()
    y_test = load_data(os.path.join(data_dir, 'y_test.csv')).squeeze().to_numpy()

    models_with_params = get_models_and_params(seed=42)
    model_summaries = []
    trained_models = {}

    for name, (model, param_grid) in models_with_params.items():
        logging.info(f"Training {name} with nested CV")
        model_path = os.path.join(models_dir, f"{name}.pkl")

        if os.path.exists(model_path):
            logging.info(f"Loading pretrained {name}")
            trained_model = load_model(name, models_dir)
        else:
            summary_df, trained_model = nested_cv_regression(model, param_grid, X_train, y_train)
            save_model(trained_model, name, models_dir)
            summary_df.to_csv(os.path.join(results_dir, f"nestedcv_{name}.csv"), index=False)

        trained_models[name] = trained_model

    # Evaluate on test data
    metrics_df, predictions = evaluate_models(trained_models, X_test, y_test)
    metrics_dict[subset_name] = metrics_df

    # Save predictions and metrics
    save_results(y_test, predictions, os.path.join(results_dir, 'predictions.csv'))
    metrics_df.to_csv(os.path.join(results_dir, 'model_metrics.csv'), index=False)

    # Build ensemble from top-k models (based on R2)
    top_k = 3
    top_models = metrics_df.sort_values(by='R2', ascending=False).head(top_k)['Model'].tolist()
    selected_models = {name: trained_models[name] for name in top_models}
    ensemble = build_voting_regressor(selected_models)
    ensemble.fit(X_train, y_train)
    save_model(ensemble, 'ensemble', models_dir)

    y_pred_ens = ensemble.predict(X_test)
    ensemble_bootstrap_df = evaluate_regression_ensemble_bootstrap(y_true=y_test, y_pred=y_pred_ens)
    ensemble_bootstrap_df.to_csv(os.path.join(results_dir, 'ensemble_metrics.csv'), index=False)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='OutputDir_LCDP_NCV', help="Main output directory")
    return parser.parse_args()


def main():
    args = parse_arguments()
    metrics_dict = {}

    feature_combinations = [
                            '1_MO', 
                            '2_MS', 
                            '3_GT', 
                            '4_MO_MS', 
                            '5_MO_MS_GT', 
                            '6_DG_MO_MS_GT'
                        ]
    subsets = ['without_nan']

    for feature_combo in feature_combinations:
        for subset_name in subsets:
            logging.info(f"Processing: {feature_combo} | Subset: {subset_name}")
            subset_dir = create_directory_structure(args.output_dir, feature_combo, subset_name)
            data_dir = os.path.join(subset_dir, 'data')
            models_dir = os.path.join(subset_dir, 'models')
            results_dir = os.path.join(subset_dir, 'results')

            train_and_evaluate_models(data_dir, models_dir, results_dir, subset_name, metrics_dict)

        save_metrics_to_excel(metrics_dict, os.path.join(args.output_dir, f'RegressionMetrics_{feature_combo}.xlsx'))



if __name__ == '__main__':
    main()
