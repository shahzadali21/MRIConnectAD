# Copyright: Shahzad Ali
# Author: Shahzad Ali
# e-mail: shahzad.ali6@unibo.it
# Created: 2025-03-21
# Last modified: 2025-04-14

"""
This module contains various functions for explainability in regression tasks.
It includes SHAP and LIME explainability plots tailored for regression.
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import shap
import joblib
from lime import lime_tabular

from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor, VotingRegressor, StackingRegressor)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, DotProduct
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

###########################################
#      SHAP Explainability Plot Functions
###########################################

def plot_shap_summary_regression(model, X_test, output_dir):
    """
    Plot SHAP summary plot for regression models.
    Supports tree-based, linear, and general models.
    """
    shap.initjs()

    # Determine the appropriate SHAP explainer
    if hasattr(model, 'tree_'):
        explainer = shap.TreeExplainer(model)  # Tree-based models
    elif isinstance(model, (LinearRegression, SGDRegressor)):
        explainer = shap.LinearExplainer(model, X_test)  # Linear models
    elif isinstance(model, SVR):
        print(f"Using KernelExplainer for SVR model: {model.__class__.__name__}")
        explainer = shap.KernelExplainer(model.predict, shap.kmeans(X_test, min(50, X_test.shape[0])))
    elif isinstance(model, AdaBoostRegressor):
        print(f"Using KernelExplainer for AdaBoostRegressor model: {model.__class__.__name__}")
        explainer = shap.KernelExplainer(model.predict, shap.kmeans(X_test, min(50, X_test.shape[0])))
    else:
        # Generic models (fallback to default SHAP explainer)
        explainer = shap.Explainer(model, X_test)

    # Calculate SHAP values
    shap_values = explainer(X_test)

    # Plot SHAP summary
    plt.figure()
    shap.summary_plot(shap_values, X_test, max_display=5, show=False)
    plt.title(f"SHAP Summary Plot - {model.__class__.__name__}")
    
    # Save the plot
    output_path = os.path.join(output_dir, f"Shap_Summary_{model.__class__.__name__}.png")
    plt.savefig(output_path, format="png", bbox_inches="tight", dpi=500)
    plt.close()
    print(f"SHAP summary plot saved to {output_dir}.")


def plot_shap_feature_importance(model, X_test, output_dir, results_dir):
    """
    Generate a SHAP-based feature importance plot and save feature importances to a CSV.
    """
    shap.initjs()

    # Determine the appropriate SHAP explainer
    if hasattr(model, 'tree_'):
        explainer = shap.TreeExplainer(model)  # Tree-based models
    elif isinstance(model, (LinearRegression, SGDRegressor)):
        explainer = shap.LinearExplainer(model, X_test)  # Linear models
    elif isinstance(model, SVR):
        print(f"Using KernelExplainer for SVR model: {model.__class__.__name__}")
        explainer = shap.KernelExplainer(model.predict, shap.kmeans(X_test, min(50, X_test.shape[0])))
    elif isinstance(model, AdaBoostRegressor):
        print(f"Using KernelExplainer for AdaBoostRegressor model: {model.__class__.__name__}")
        explainer = shap.KernelExplainer(model.predict, shap.kmeans(X_test, min(50, X_test.shape[0])))
    else:
        # Generic models (fallback to default SHAP explainer)
        explainer = shap.Explainer(model, X_test)

    # Calculate SHAP values
    shap_values = explainer(X_test)

    # Compute mean absolute SHAP values
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)

    # Create a DataFrame to store the feature importances
    shap_importances = pd.DataFrame({
        "Feature": X_test.columns,
        "Importance": mean_abs_shap
    }).sort_values(by="Importance", ascending=False)

    # Save SHAP feature importances to a CSV file
    csv_path = os.path.join(results_dir, f'Shap_Feature_Importance_{model.__class__.__name__}.csv')
    shap_importances.to_csv(csv_path, index=False)
    print(f"SHAP Feature Importance saved to {csv_path}")
    
    # Plot SHAP feature importance
    plt.figure(figsize=(12, 8))
    sns.barplot(x="Importance", y="Feature", data=shap_importances.head(15), palette="coolwarm")
    plt.xlabel("Mean SHAP Value")
    plt.ylabel("Feature")
    plt.title("SHAP Feature Importances")
    plt.tight_layout()

    plt.savefig(f'{output_dir}/Shap_Feature_Importance_{model.__class__.__name__}.png', format='png', bbox_inches='tight', dpi=500)
    plt.close()
    print(f"SHAP feature importance plot saved to {output_dir}.")


def plot_shap_waterfall_regression(model, X_test, output_dir, sample_index=0):
    """
    Plot SHAP waterfall plot for a specific sample in regression models.
    """
    shap.initjs()

    # Determine the appropriate SHAP explainer
    if hasattr(model, 'tree_'):
        explainer = shap.TreeExplainer(model)  # Tree-based models
    elif isinstance(model, (LinearRegression, SGDRegressor)):
        explainer = shap.LinearExplainer(model, X_test)  # Linear models
    elif isinstance(model, SVR):
        print(f"Using KernelExplainer for SVR model: {model.__class__.__name__}")
        explainer = shap.KernelExplainer(model.predict, shap.kmeans(X_test, min(50, X_test.shape[0])))
    elif isinstance(model, AdaBoostRegressor):
        print(f"Using KernelExplainer for AdaBoostRegressor model: {model.__class__.__name__}")
        explainer = shap.KernelExplainer(model.predict, shap.kmeans(X_test, min(50, X_test.shape[0])))
    else:
        # Generic models (fallback to default SHAP explainer)
        explainer = shap.Explainer(model, X_test)

    # Calculate SHAP values
    shap_values = explainer(X_test)

    # Generate a waterfall plot for a specific sample
    shap.waterfall_plot(
        shap.Explanation(values=shap_values[sample_index].values,
                         base_values=shap_values[sample_index].base_values,
                         data=X_test.iloc[sample_index]),
        show=False
    )
    plt.title(f"SHAP Waterfall Plot - Sample {sample_index}")

    # Save the plot
    output_path = os.path.join(output_dir, f"Shap_Waterfall_Sample_{sample_index}_{model.__class__.__name__}.png")
    plt.savefig(output_path, format="png", bbox_inches="tight", dpi=500)
    plt.close()

    print(f"SHAP waterfall plot for sample {sample_index} saved to {output_dir}.")



###########################################
#      LIME Explainability Plot Functions
###########################################

def plot_lime_explanation_regression(model, X_train, X_test, sample_index, num_features, output_dir):
    """
    Generate and save LIME explanation for a specific instance in regression tasks.
    """
    explainer = lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X_train.columns,
        discretize_continuous=True,
        mode='regression'
    )

    exp = explainer.explain_instance(
        data_row=X_test.iloc[sample_index].values,
        predict_fn=model.predict,
        num_features=num_features
    )
    exp.save_to_file(f"{output_dir}/lime_explanation_{model.__class__.__name__}.html")
    joblib.dump(exp, f"{output_dir}/lime_explanation_{model.__class__.__name__}.pkl")
    print(f"LIME explanation saved to {output_dir}.")

###########################################
