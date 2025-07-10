# Copyright: Shahzad Ali
# Author: Shahzad Ali
# e-mail: shahzad.ali6@unibo.it
# Created: 2025-03-21
# Last modified: 2025-04-14

"""
This module contains various functions related to visualizations for regression models.
It includes functions for plotting feature importance, model performance comparison,
RMSE and R² plots, and SHAP and LIME explainability.
"""

import os
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



def plot_model_performance_comparison(metrics, model_names, output_dir, filename_prefix="Regression_ModelEvaluationMetrics"):
    """ Plots a comparison of evaluation metrics (MAE, MSE, RMSE, R²) for regression models. """
    plt.figure(figsize=(12, 8))
    ax = metrics.plot(kind='bar', edgecolor='black')
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.title("Comparison of Regression Metrics for All Models", fontsize=14)
    # Set custom x-axis labels to model names
    ax.set_xticklabels(model_names)
    #plt.gca().set_xticklabels(model_names)
    
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Metrics')
    plt.grid(axis='y', linestyle='--', linewidth=0.5, color='gray')

    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{filename_prefix}.png', format='png', bbox_inches='tight', dpi=500)
    plt.close()
    logging.info(f"Comparison of regression metrics saved to {output_dir}.")


def plot_rmse(metrics, model_names, output_dir, filename_prefix="Regression_Model_MAE"):
    """ Plots MAE comparison for all models. """
    plt.figure(figsize=(10, 6))
    sns.barplot(x=model_names, y=metrics['MAE'], palette="Blues")
    plt.xlabel("Model", fontsize=12)
    plt.ylabel("MAE", fontsize=12)
    plt.title("Mean Absolute Error (MAE) for All Models", fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)

    # Annotate RMSE values
    for i, value in enumerate(metrics['RMSE']):
        plt.text(i, value, f"{value:.2f}", ha='center', va='bottom', fontsize=10)

    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{filename_prefix}.png', format='png', bbox_inches='tight', dpi=500)
    plt.close()
    logging.info(f"RMSE comparison plot saved to {output_dir}.")


def plot_r2(metrics, model_names, output_dir, filename_prefix="Regression_Model_R2"):
    """ Plots R² comparison for all models. """
    plt.figure(figsize=(10, 6))
    sns.barplot(x=model_names, y=metrics['R2'], palette="Greens")
    plt.xlabel("Model", fontsize=12)
    plt.ylabel("R²", fontsize=12)
    plt.title("R² for All Models", fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)

    # Annotate R² values
    for i, value in enumerate(metrics['R2']):
        plt.text(i, value, f"{value:.2f}", ha='center', va='bottom', fontsize=10)

    filename = f"{filename_prefix}.pdf"
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{filename_prefix}.png', format='png', bbox_inches='tight', dpi=500)
    plt.close()
    logging.info(f"R² comparison plot saved to {output_dir}/{filename}.")


def plot_feature_importance(model, feature_names, top_n, results_dir, output_dir):
    """
    Plots the top N feature importances for a regression model and saves feature importances to a CSV file.
    """
    if not hasattr(model, 'feature_importances_'):
        logging.warning(f"Model {model.__class__.__name__} does not support feature importances.")
        return

    feature_importances = model.feature_importances_
    indices = np.argsort(feature_importances)[::-1][:top_n]  # Top N features

    # Create DataFrame for top features
    top_features_df = pd.DataFrame(
        [(feature_names[i], feature_importances[i]) for i in indices],
        columns=['Feature', 'Importance']
    )

    # Save feature importances to a CSV file
    os.makedirs(results_dir, exist_ok=True)
    csv_path = f"{results_dir}/Feature_Importance_{model.__class__.__name__}.csv"
    top_features_df.to_csv(csv_path, index=False)
    logging.info(f"Feature importance data saved to {csv_path}.")

    # Plot feature importances
    plt.figure(figsize=(12, 8))
    sns.barplot(x="Feature", y="Importance", data=top_features_df, palette="viridis")
    plt.xlabel("Feature")
    plt.ylabel("Importance Score")
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.title(f"Top {top_n} Important Feature Identified by Random Forest")
    plt.tight_layout()

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plot_path = f"{output_dir}/Feature_Importance_{model.__class__.__name__}.png"
    plt.savefig(plot_path, format='png', bbox_inches='tight', dpi=500)
    plt.close()
    logging.info(f"Feature importance plot saved to {plot_path}.")
