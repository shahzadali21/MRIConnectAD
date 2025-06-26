# Copyright: Shahzad Ali
# Author: Shahzad Ali
# e-mail: shahzad.ali6@unibo.it
# Created: 2024-10-24
# Last modified: 2024-10-24

"""
This module contains various functions related to visualizations.
It includes functions for plotting decision trees, model performance comparison,
accuracy comparison, confusion matrices, and SHAP and LIME explainability plots.
"""

import os
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.tree import plot_tree

import shap
import joblib
from lime import lime_tabular

from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression, LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier




###########################################
#      Visualization Plot Functions
###########################################
def plot_decision_tree(model, feature_names, output_dir):
    plt.figure(figsize=(12, 8))
    plot_tree(model, feature_names=feature_names, filled=True, rounded=True, proportion=False)
    plt.title("Decision Tree")
    #plt.savefig(f'{output_dir}/Plot_DecisionTree.pdf', format='pdf')
    plt.savefig(f'{output_dir}/Plot_DecisionTree.png', format='png', bbox_inches='tight', dpi=500)
    plt.close()
    print(f"Decision Tree is saved to {output_dir}.")


def plot_feature_importance(model, feature_names, top_n, results_dir, output_dir):
    """ 
    Plot the top N feature importances using a trained model and save the importances to a CSV file.
    """
    feature_importances = model.feature_importances_       # Get feature importances
    #feature_names = list(feature_names)                   # Get the feature names
    indices = np.argsort(feature_importances)[::-1]        # Sort indices of features by importance
    
    # Extract top N features and their importances
    top_features = [(feature_names[i], feature_importances[i]) for i in indices[:top_n]]
    top_features_df = pd.DataFrame(top_features, columns=['Feature', 'Importance'])

    # Save the top feature importances to a CSV file
    csv_path = os.path.join(results_dir, f'Top_{top_n}_Feature_Importances_{model.__class__.__name__}.csv')
    top_features_df.to_csv(csv_path, index=False)
    print(f"Top {top_n} feature importances saved to {csv_path}.")

    plt.figure(figsize=(12, 8))  # Increase figure size for readability
    sns.barplot(x='Feature', y='Importance', data=top_features_df, palette='viridis')
    plt.xlabel('Feature')
    plt.ylabel('Importance Score')
    plt.xticks(rotation=60, ha='right')  # Rotate labels more and align to the right
    plt.title(f'Top {top_n} Feature Importances using {model.__class__.__name__}')
    plt.tight_layout()

    # Save the plot
    plot_path = os.path.join(output_dir, f'Feature_Importance_by_{model.__class__.__name__}.png')
    plt.savefig(plot_path, format='png', bbox_inches='tight', dpi=500)
    plt.close()
    print(f"Feature Importances plot saved to {plot_path}.")


def plot_model_performance_comparison(metrics, model_names, output_dir, filename_prefix="Comparison_ModelEvaluationMetrics"):
    """    Plots a comparison of evaluation metrics for all models.    """
    plt.figure(figsize=(12, 8))
    ax = metrics.plot(kind='bar', edgecolor='black')
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.title("Comparison of Evaluation Metrics of all Models", fontsize=14)
    # Set custom x-axis labels to model names
    ax.set_xticklabels(model_names)

    # Add legend outside of the plot
    legend_labels = metrics.columns
    ax.legend(legend_labels, loc='center left', bbox_to_anchor=(1, 0.5), title='Metrics')
    
    # Add gridlines for better readability
    plt.grid(which='major', linestyle='--', linewidth=0.5, color='gray')
    plt.minorticks_on()
    plt.grid(which='minor', linestyle=':', linewidth=0.5, color='lightgray')

    # Set y-axis limit if necessary (optional adjustment based on data)
    plt.ylim(0, 100)   # plt.ylim(0, metrics.max().max() * 1.1)
    # Adjust layout to prevent clipping of labels
    plt.tight_layout()
    
    # Save the plot
    plt.title("Comparison of Evaluation Metrics of all Models")
    plt.savefig(f'{output_dir}/{filename_prefix}.png', format='png', bbox_inches='tight', dpi=500)
    plt.close()


def plot_model_accuracy(metrics, model_names, output_dir, filename_prefix="Comparison_ModelAccuracy"):
    """ Plots a comparison of model accuracy for all models. """
    plt.figure(figsize=(10, 5))
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Model Accuracy', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)


    models_score = metrics['Accuracy'].tolist()
    ax = sns.barplot(x=model_names, y=models_score, palette="YlGnBu")
    for i, v in enumerate(models_score):
        ax.text(i, v, str(v), ha='center', va='bottom')

    plt.tight_layout()
    
    # Save the plot
    plt.title("Model Comparison - Model Accuracy")
    #plt.savefig(f'{output_dir}/{filename_prefix}.pdf', format='pdf', bbox_inches='tight')
    plt.savefig(f'{output_dir}/{filename_prefix}.png', format='png', bbox_inches='tight', dpi=500)
    plt.close()


###########################################
#      SHAP Explainability Plot Functions
###########################################

def plot_shap_summary_by_class(model, X_test, class_mapping, output_dir):
    """
    Plot SHAP summary plots for a variety of models.
    Supports tree-based, linear, and general models.
    """
    shap.initjs()

    # Handle ensemble models
    if hasattr(model, "estimators_") or isinstance(model, (VotingClassifier, StackingClassifier)):
        # Ensemble model handling
        print(f"Detected ensemble model: {model.__class__.__name__}")
        for idx, sub_model in enumerate(model.estimators_):
            print(f"Processing sub-model {idx + 1}/{len(model.estimators_)}: {sub_model.__class__.__name__}")
            plot_shap_summary_by_class(sub_model, X_test, class_mapping, output_dir)  # Recursive call
        return  # Exit after processing all sub-models
    
    # Determine the appropriate SHAP explainer based on the model type
    if hasattr(model, 'tree_'):
        # Tree-based models (e.g., RandomForest, GradientBoosting, XGBoost)
        explainer = shap.TreeExplainer(model)
    elif isinstance(model, (LogisticRegression, SGDClassifier, LinearDiscriminantAnalysis)):
        # Linear models (e.g., Logistic Regression, SGDClassifier, LDA)
        explainer = shap.LinearExplainer(model, X_test)
    elif isinstance(model, (SVC, GaussianNB, KNeighborsClassifier)):
        # Kernel or non-linear models (e.g., SVM, Naive Bayes, KNN)
        if hasattr(model, 'predict_proba'):
            explainer = shap.KernelExplainer(model.predict_proba, shap.kmeans(X_test, min(50, X_test.shape[0])))
        else:
            raise ValueError(f"SVM models without 'probability=True' are not supported. Please enable probability estimates.")
    elif isinstance(model, (XGBClassifier, GradientBoostingClassifier, AdaBoostClassifier)):
        # Boosting models
        explainer = shap.TreeExplainer(model)
    else:
        # Generic models (fallback to default SHAP explainer)
        explainer = shap.Explainer(model, X_test)

    # Generate SHAP values
    shap_values = explainer(X_test)

    # Handle binary and multiclass cases
    if isinstance(shap_values, list):  # Multiclass case: SHAP returns a list of arrays
        num_classes = len(shap_values)
        shap_values_list = [sv.values for sv in shap_values]  # Extract values for each class
    elif len(shap_values.values.shape) == 2:  # Binary classification or single-output
        num_classes = 1
        shap_values_list = [shap_values.values, -shap_values.values]  # Add SHAP values for class 0
        num_classes = 2
    else:  # Multiclass with a single SHAP array
        num_classes = shap_values.values.shape[2]
        shap_values_list = [shap_values[..., class_index].values for class_index in range(num_classes)]

    # Generate SHAP summary plots for each class
    for class_index in range(num_classes):
        class_shap_values = shap_values_list[class_index]
        class_name = class_mapping.get(class_index, f"Class_{class_index}")

        if not isinstance(class_shap_values, np.ndarray):
            class_shap_values = np.array(class_shap_values)
        
        # Debugging info
        print(f"Generating SHAP plot for class '{class_name}' with shape: {class_shap_values.shape}")


        # Convert to DataFrame
        shap_df = pd.DataFrame(class_shap_values, columns=X_test.columns)

        # Calculate mean absolute SHAP values
        mean_abs_shap = shap_df.abs().mean().sort_values(ascending=False)
        class_name = class_mapping[class_index] if num_classes > 1 else list(class_mapping.values())[0]
        print(f"Class {class_name} mean absolute SHAP values:")
        #print(mean_abs_shap)

        # Plot SHAP summary
        plt.figure()
        shap.summary_plot(class_shap_values, X_test, max_display=5, show=False)
        plt.title(f'SHAP Summary Plot - Class {class_name}')

        # Save the plot
        output_path = os.path.join(output_dir, f'Shap_summary_plot_class_{class_name}_{model.__class__.__name__}.png')
        plt.savefig(output_path, format='png', bbox_inches='tight', dpi=500)
        plt.close()

    print(f"SHAP summary plots for each class have been saved to {output_dir}.")


def plot_shap_aggregated_summary(model, X_test, output_dir):
    """
    Plot aggregated SHAP summary plot for a variety of models.
    Supports tree-based, linear, and general models.
    """
    shap.initjs()

    # Handle ensemble models
    if hasattr(model, "estimators_") or isinstance(model, (VotingClassifier, StackingClassifier)):
        # Ensemble model handling
        print(f"Detected ensemble model: {model.__class__.__name__}")
        for idx, sub_model in enumerate(model.estimators_):
            print(f"Processing sub-model {idx + 1}/{len(model.estimators_)}: {sub_model.__class__.__name__}")
            plot_shap_aggregated_summary(sub_model, X_test, output_dir)  # Recursive call
        return  # Exit after processing all sub-models

    # Determine the appropriate SHAP explainer based on the model type
    if hasattr(model, 'tree_'):
        explainer = shap.TreeExplainer(model)
    elif isinstance(model, (LogisticRegression, SGDClassifier, LinearDiscriminantAnalysis)):
        explainer = shap.LinearExplainer(model, X_test)
    elif isinstance(model, (SVC, GaussianNB, KNeighborsClassifier)):
        if hasattr(model, 'predict_proba'):
            explainer = shap.KernelExplainer(model.predict_proba, shap.kmeans(X_test, min(50, X_test.shape[0])))
        else:
            raise ValueError("SVM models without 'probability=True' are not supported. Please enable probability estimates.")
    elif isinstance(model, (XGBClassifier, GradientBoostingClassifier, AdaBoostClassifier)):
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.Explainer(model, X_test)

    # Generate SHAP values
    shap_values = explainer(X_test)

    # Aggregate SHAP values across classes if multi-class
    if isinstance(shap_values, list):  # Multi-class case: SHAP returns a list of arrays
        aggregated_shap_values = np.mean([np.abs(sv.values) for sv in shap_values], axis=0)  # Average across classes
    elif len(shap_values.values.shape) == 3:  # Multi-class with a single SHAP array
        aggregated_shap_values = np.abs(shap_values.values).mean(axis=2)  # Average across class dimension
    elif len(shap_values.values.shape) == 2:  # Binary classification or regression
        aggregated_shap_values = np.abs(shap_values.values)  # Directly use SHAP values
    else:
        raise ValueError(f"Unexpected SHAP value dimensions: {shap_values.values.shape}")

    # Convert to DataFrame for aggregated SHAP values
    aggregated_shap_df = pd.DataFrame(aggregated_shap_values, columns=X_test.columns)

    # Calculate mean absolute SHAP values for all features
    mean_abs_shap = aggregated_shap_df.abs().mean().sort_values(ascending=False)
    print("Mean absolute aggregated SHAP values:")
    print(mean_abs_shap)

    # Plot aggregated SHAP summary
    plt.figure()
    shap.summary_plot(aggregated_shap_values, X_test, max_display=5, show=False)
    plt.title(f'SHAP Summary Plot - Aggregated')
    
    # Save the plot
    output_path = os.path.join(output_dir, f'Shap_summary_plot_Aggregated_{model.__class__.__name__}.png')
    plt.savefig(output_path, format='png', bbox_inches='tight', dpi=500)
    plt.close()

    print(f"Aggregated SHAP summary plot has been saved to {output_dir}.")


def plot_shap_dependence_and_feature_importance(model, X_test, class_mapping, output_dir, results_dir):
    """
    Plot SHAP dependence plots for all features and compare feature importance across all classes.
    Handles tree-based, linear, kernel-based, and ensemble models with binary, multi-class, or regression tasks.
    """
    shap.initjs()

    # Handle ensemble models (e.g., VotingClassifier, StackingClassifier)
    if hasattr(model, "estimators_") or isinstance(model, (VotingClassifier, StackingClassifier)):
        print(f"Detected ensemble model: {model.__class__.__name__}")
        for idx, sub_model in enumerate(model.estimators_):
            print(f"Processing sub-model {idx + 1}/{len(model.estimators_)}: {sub_model.__class__.__name__}")
            plot_shap_dependence_and_feature_importance(sub_model, X_test, class_mapping, output_dir, results_dir)
        return  # Exit after processing all sub-models

    # Determine which SHAP explainer to use based on the model type
    if hasattr(model, 'tree_'):
        explainer = shap.TreeExplainer(model)  # Use TreeExplainer for tree-based models
    elif isinstance(model, (LogisticRegression, SGDClassifier, LinearRegression)):
        explainer = shap.LinearExplainer(model, X_test)  # Use LinearExplainer for linear models
    elif isinstance(model, (SVC, GaussianNB, KNeighborsClassifier)):
        if hasattr(model, 'predict_proba'):
            explainer = shap.KernelExplainer(model.predict_proba, shap.kmeans(X_test, min(50, X_test.shape[0])))
        else:
            raise ValueError("SVM models without 'probability=True' are not supported. Please enable probability estimates.")
    elif isinstance(model, (XGBClassifier, GradientBoostingClassifier, AdaBoostClassifier)):
        explainer = shap.TreeExplainer(model)  # Use TreeExplainer for boosting models
    else:
        explainer = shap.Explainer(model, X_test)  # Fallback to the general SHAP explainer

    # Calculate SHAP values
    shap_values = explainer(X_test)

    # Determine the number of classes and handle SHAP values
    if hasattr(shap_values, 'values'):
        if len(shap_values.values.shape) == 2:  # Binary classification or regression
            shap_values_list = [shap_values.values]
            num_classes = 1
        elif len(shap_values.values.shape) == 3:  # Multi-class classification
            num_classes = shap_values.values.shape[2]
            shap_values_list = [shap_values[..., class_idx].values for class_idx in range(num_classes)]
    elif isinstance(shap_values, list):  # Multi-class classification (list format)
        num_classes = len(shap_values)
        shap_values_list = [sv.values for sv in shap_values]
    else:
        raise ValueError("Unexpected SHAP values structure. Cannot determine class dimension.")

    # Create a DataFrame to store feature importance
    feature_importance = pd.DataFrame()

    # Compute feature importance for each class
    for class_index, class_shap_values in enumerate(shap_values_list):
        class_name = class_mapping[class_index] if num_classes > 1 else list(class_mapping.values())[0]
        shap_df = pd.DataFrame(class_shap_values, columns=X_test.columns)

        mean_abs_shap = shap_df.abs().mean().sort_values(ascending=False)
        feature_importance[f'Class_{class_name}'] = mean_abs_shap

    # Save feature importance to CSV
    csv_path = os.path.join(results_dir, f'Shap_Feature_Importance_{model.__class__.__name__}.csv')
    feature_importance.index.name = 'Feature'
    feature_importance.to_csv(csv_path)
    print(f"SHAP Feature Importance saved to {csv_path}_{model.__class__.__name__}")

    """
    # Generate dependence plots for each feature and class
    for class_index, class_shap_values in enumerate(shap_values_list):
        class_name = class_mapping[class_index] if num_classes > 1 else list(class_mapping.values())[0]
        for feature in X_test.columns:
            plt.figure()
            shap.dependence_plot(feature, class_shap_values, X_test, show=False)
            plt.title(f'Dependence Plot - Class {class_name} - Feature: {feature}')
            pdf_path = os.path.join(output_dir, f'Shap_dependence_plot_class_{class_name}_{feature}.pdf')
            plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
            plt.close()

    print(f"SHAP dependence plots for all features and classes saved to {output_dir}.")
    """


def plot_decision_for_all_samples_by_class(model, X_test, y_test, class_mapping, output_dir):
    """
    Generate SHAP decision plots for all samples combined, grouped by class.
    Supports tree-based, linear, ensemble, and general models.
    """

    shap.initjs()

    # Handle ensemble models (e.g., VotingClassifier, StackingClassifier)
    if hasattr(model, "estimators_") or isinstance(model, (VotingClassifier, StackingClassifier)):
        print(f"Detected ensemble model: {model.__class__.__name__}")
        for idx, sub_model in enumerate(model.estimators_):
            print(f"Processing sub-model {idx + 1}/{len(model.estimators_)}: {sub_model.__class__.__name__}")
            plot_decision_for_all_samples_by_class(sub_model, X_test, y_test, class_mapping, output_dir)
        return

    # Determine the appropriate SHAP explainer
    if hasattr(model, 'tree_'):
        explainer = shap.TreeExplainer(model)  # Tree-based models
    elif isinstance(model, (LogisticRegression, SGDClassifier, LinearDiscriminantAnalysis)):
        explainer = shap.LinearExplainer(model, X_test)  # Linear models
    elif isinstance(model, (SVC, GaussianNB, KNeighborsClassifier)):
        if hasattr(model, 'predict_proba'):
            explainer = shap.KernelExplainer(model.predict_proba, shap.kmeans(X_test, min(50, X_test.shape[0])))
        else:
            raise ValueError(f"SVM models without 'probability=True' are not supported.")
    elif isinstance(model, (XGBClassifier, GradientBoostingClassifier, AdaBoostClassifier)):
        explainer = shap.TreeExplainer(model)  # Boosting models
    else:
        explainer = shap.Explainer(model, X_test)  # General models

    # Generate SHAP values
    shap_values = explainer.shap_values(X_test)
    feature_names = X_test.columns.tolist()

    # Handle binary and multiclass cases
    if isinstance(shap_values, list):  # Multiclass case: SHAP returns a list of arrays
        num_classes = len(shap_values)
        shap_values_list = shap_values
    elif len(shap_values.shape) == 2:  # Binary classification or single-output
        num_classes = 2  # Treat as two classes (positive and negative)
        shap_values_list = [shap_values, -shap_values]  # Positive and negative classes
    else:  # Multiclass with a single SHAP array
        num_classes = shap_values.shape[2]
        shap_values_list = [shap_values[:, :, class_index] for class_index in range(num_classes)]

    # Generate decision plots for each class
    for class_index in range(num_classes):
        class_shap_values = shap_values_list[class_index]

        # Filter samples for the current class
        class_samples_mask = (y_test == class_index)
        if not class_samples_mask.any():
            print(f"No samples found for class {class_mapping[class_index]}. Skipping.")
            continue

        class_samples = X_test[class_samples_mask]
        class_shap_values = class_shap_values[class_samples_mask]

        # Generate the correct base value for multiclass or binary
        if isinstance(explainer.expected_value, list):  # Multiclass case
            base_value = explainer.expected_value[class_index]
        else:  # Binary or single-output case
            base_value = explainer.expected_value

        # Ensure base_value is scalar or of compatible shape
        if isinstance(base_value, np.ndarray) and len(base_value) > 1:
            base_value = base_value[class_index]

        # Generate the decision plot
        shap.decision_plot(
            base_value=base_value,
            shap_values=class_shap_values,
            features=class_samples.values,
            feature_names=feature_names,
            legend_location="lower right",
            show=False
        )
        plt.title(f"SHAP Decision Plot - All Samples (Class {class_mapping[class_index]})")

        # Save the plot
        output_path = f"{output_dir}/Shap_Decision_Plot_Class_{class_mapping[class_index]}_{model.__class__.__name__}.png"
        plt.savefig(output_path, format="png", bbox_inches="tight", dpi=500)
        plt.close()

    print(f"Class-specific SHAP decision plots for all samples have been saved to {output_dir}.")

######################################################
#      NEW SHAP Plot Functions (need to be modified)
######################################################
def plot_waterfall_aggregated_by_class(model, X_test, y_test, class_mapping, output_dir):
    """
    Generate aggregated SHAP waterfall plots for all samples in a class.
    Handles binary and multiclass classification scenarios.
    """
    shap.initjs()

    # Handle ensemble models (e.g., VotingClassifier, StackingClassifier)
    if hasattr(model, "estimators_") or isinstance(model, (VotingClassifier, StackingClassifier)):
        print(f"Detected ensemble model: {model.__class__.__name__}")
        for idx, sub_model in enumerate(model.estimators_):
            print(f"Processing sub-model {idx + 1}/{len(model.estimators_)}: {sub_model.__class__.__name__}")
            plot_waterfall_aggregated_by_class(sub_model, X_test, y_test, class_mapping, output_dir)
        return

    # Determine the appropriate SHAP explainer
    if hasattr(model, 'tree_'):
        explainer = shap.TreeExplainer(model)  # Tree-based models
    elif isinstance(model, (LogisticRegression, SGDClassifier, LinearDiscriminantAnalysis)):
        explainer = shap.LinearExplainer(model, X_test)  # Linear models
    elif isinstance(model, (SVC, GaussianNB, KNeighborsClassifier)):
        if hasattr(model, 'predict_proba'):
            explainer = shap.KernelExplainer(model.predict_proba, shap.kmeans(X_test, min(50, X_test.shape[0])))
        else:
            raise ValueError(f"SVM models without 'probability=True' are not supported.")
    elif isinstance(model, (XGBClassifier, GradientBoostingClassifier, AdaBoostClassifier)):
        explainer = shap.TreeExplainer(model)  # Boosting models
    else:
        explainer = shap.Explainer(model, X_test)  # General models

    # Generate SHAP values
    shap_values = explainer(X_test)

    # Handle binary and multiclass cases
    if isinstance(shap_values.values, list):  # Multiclass case: SHAP returns a list of arrays
        num_classes = len(shap_values.values)
        shap_values_list = shap_values.values
    elif len(shap_values.values.shape) == 2:  # Binary classification or single-output
        num_classes = 2  # Binary case
        shap_values_list = [shap_values.values, -shap_values.values]
    else:  # Multiclass with a single SHAP array
        num_classes = shap_values.values.shape[2]
        shap_values_list = [shap_values[..., class_index].values for class_index in range(num_classes)]

    # Generate aggregated waterfall plots for each class
    for class_index in range(num_classes):
        # Filter samples for the current class
        class_samples_mask = (y_test == class_index)
        if not class_samples_mask.any():
            print(f"No samples found for class {class_mapping[class_index]}. Skipping.")
            continue

        class_samples = X_test[class_samples_mask]
        class_shap_values = shap_values_list[class_index][class_samples_mask]

        # Aggregate SHAP values
        mean_shap_values = np.mean(class_shap_values, axis=0)
        aggregated_features = class_samples.mean(axis=0)

        # Extract the base value for the current class
        if isinstance(shap_values.base_values, np.ndarray) and shap_values.base_values.ndim > 1:
            base_value = shap_values.base_values[class_samples_mask, class_index].mean()
        elif isinstance(shap_values.base_values, np.ndarray):
            base_value = shap_values.base_values[class_samples_mask].mean()
        else:
            base_value = shap_values.base_values

        # Generate the aggregated waterfall plot
        shap.waterfall_plot(
            shap.Explanation(
                values=mean_shap_values,
                base_values=base_value,
                data=aggregated_features.values,
                feature_names=aggregated_features.index.tolist(),
            ), show=False
        )

        plt.title(f"Aggregated SHAP Waterfall Plot - Class {class_mapping[class_index]}")

        # Save the plot
        output_path = f"{output_dir}/Shap_Waterfall_Plot_Aggregated_Class_{class_mapping[class_index]}_{model.__class__.__name__}.png"
        plt.savefig(output_path, format="png", bbox_inches="tight", dpi=500)
        plt.close()

    print(f"Aggregated SHAP waterfall plots for all classes have been saved to {output_dir}.")


def plot_waterfall_for_all_classes_combined(model, X_test, y_test, class_mapping, output_dir):
    """
    Generate a single SHAP aggregated waterfall plot for all classes together.
    Handles binary and multiclass scenarios.
    """
    import shap
    import matplotlib.pyplot as plt
    import numpy as np

    shap.initjs()

    # Handle ensemble models
    if hasattr(model, "estimators_") or isinstance(model, (VotingClassifier, StackingClassifier)):
        print(f"Detected ensemble model: {model.__class__.__name__}")
        for idx, sub_model in enumerate(model.estimators_):
            print(f"Processing sub-model {idx + 1}/{len(model.estimators_)}: {sub_model.__class__.__name__}")
            plot_waterfall_for_all_classes_combined(sub_model, X_test, y_test, class_mapping, output_dir)
        return

    # Determine the appropriate SHAP explainer
    if hasattr(model, 'tree_'):
        explainer = shap.TreeExplainer(model)  # Tree-based models
    elif isinstance(model, (LogisticRegression, SGDClassifier, LinearDiscriminantAnalysis)):
        explainer = shap.LinearExplainer(model, X_test)  # Linear models
    elif isinstance(model, (SVC, GaussianNB, KNeighborsClassifier)):
        if hasattr(model, 'predict_proba'):
            explainer = shap.KernelExplainer(model.predict_proba, shap.kmeans(X_test, min(50, X_test.shape[0])))
        else:
            raise ValueError(f"SVM models without 'probability=True' are not supported.")
    elif isinstance(model, (XGBClassifier, GradientBoostingClassifier, AdaBoostClassifier)):
        explainer = shap.TreeExplainer(model)  # Boosting models
    else:
        explainer = shap.Explainer(model, X_test)  # General models

    # Generate SHAP values
    shap_values = explainer(X_test)

    # Handle binary and multiclass cases
    if isinstance(shap_values.values, list):  # Multiclass case: SHAP returns a list of arrays
        num_classes = len(shap_values.values)
        shap_values_list = shap_values.values
    elif len(shap_values.values.shape) == 2:  # Binary classification or single-output
        num_classes = 2  # Binary case
        shap_values_list = [shap_values.values, -shap_values.values]
    else:  # Multiclass with a single SHAP array
        num_classes = shap_values.values.shape[2]
        shap_values_list = [shap_values[..., class_index].values for class_index in range(num_classes)]

    # Combine SHAP values across all classes
    feature_names = X_test.columns
    aggregated_shap_values = np.zeros(len(feature_names))
    base_value = 0  # Placeholder for aggregated base value

    for class_index in range(num_classes):
        class_samples_mask = (y_test == class_index)
        if not class_samples_mask.any():
            print(f"No samples found for class {class_mapping[class_index]}. Skipping.")
            continue

        class_samples = X_test[class_samples_mask]
        class_shap_values = shap_values_list[class_index][class_samples_mask]

        # Compute mean SHAP values for features
        aggregated_shap_values += class_shap_values.mean(axis=0)
        if isinstance(shap_values.base_values, np.ndarray) and shap_values.base_values.ndim > 1:
            base_value += shap_values.base_values[class_samples_mask, class_index].mean()
        else:
            base_value += shap_values.base_values[class_index] if num_classes > 1 else shap_values.base_values

    # Sort features by importance
    sorted_indices = np.argsort(np.abs(aggregated_shap_values))[::-1]
    sorted_feature_names = feature_names[sorted_indices]
    sorted_aggregated_shap_values = aggregated_shap_values[sorted_indices]

    # Generate the aggregated waterfall plot
    shap.waterfall_plot(
        shap.Explanation(
            values=sorted_aggregated_shap_values,
            base_values=base_value / num_classes,  # Normalize base value for all classes
            data=None,  # No specific sample data for aggregated plot
            feature_names=sorted_feature_names.tolist(),
        ),
        show=False
    )

    plt.title(f"SHAP Aggregated Waterfall Plot - All Classes Combined")
    plt.tight_layout()

    # Save the plot
    output_path = os.path.join(output_dir, f"Shap_Waterfall_Plot_All_Classes_Combined_{model.__class__.__name__}.png")
    plt.savefig(output_path, format="png", bbox_inches="tight", dpi=500)
    plt.close()

    print(f"Aggregated SHAP waterfall plot for all classes has been saved to {output_dir}.")


def plot_beeswarm_for_all_classes(model, X_test, y_test, class_mapping, output_dir):
    """
    Generate SHAP beeswarm plots for all samples, grouped by class.
    Handles binary and multiclass classification scenarios.
    """
    shap.initjs()

    # Handle ensemble models
    if hasattr(model, "estimators_") or isinstance(model, (VotingClassifier, StackingClassifier)):
        print(f"Detected ensemble model: {model.__class__.__name__}")
        for idx, sub_model in enumerate(model.estimators_):
            print(f"Processing sub-model {idx + 1}/{len(model.estimators_)}: {sub_model.__class__.__name__}")
            plot_beeswarm_for_all_classes(sub_model, X_test, y_test, class_mapping, output_dir)
        return

    # Determine the appropriate SHAP explainer
    if hasattr(model, 'tree_'):
        explainer = shap.TreeExplainer(model)  # Tree-based models
    elif isinstance(model, (LogisticRegression, SGDClassifier, LinearDiscriminantAnalysis)):
        explainer = shap.LinearExplainer(model, X_test)  # Linear models
    elif isinstance(model, (SVC, GaussianNB, KNeighborsClassifier)):
        if hasattr(model, 'predict_proba'):
            explainer = shap.KernelExplainer(model.predict_proba, shap.kmeans(X_test, min(50, X_test.shape[0])))
        else:
            raise ValueError(f"SVM models without 'probability=True' are not supported.")
    elif isinstance(model, (XGBClassifier, GradientBoostingClassifier, AdaBoostClassifier)):
        explainer = shap.TreeExplainer(model)  # Boosting models
    else:
        explainer = shap.Explainer(model, X_test)  # General models

    # Generate SHAP values
    shap_values = explainer(X_test)

    # Handle binary and multiclass cases
    if isinstance(shap_values.values, list):  # Multiclass case: SHAP returns a list of arrays
        num_classes = len(shap_values.values)
        shap_values_list = shap_values.values
    elif len(shap_values.values.shape) == 2:  # Binary classification or single-output
        num_classes = 2  # Binary case
        shap_values_list = [shap_values.values, -shap_values.values]
    else:  # Multiclass with a single SHAP array
        num_classes = shap_values.values.shape[2]
        shap_values_list = [shap_values[..., class_index].values for class_index in range(num_classes)]

    # Generate beeswarm plots for each class
    for class_index in range(num_classes):
        # Filter samples for the current class
        class_samples_mask = (y_test == class_index)
        if not class_samples_mask.any():
            print(f"No samples found for class {class_mapping[class_index]}. Skipping.")
            continue

        class_samples = X_test[class_samples_mask]
        class_shap_values = shap_values_list[class_index][class_samples_mask]

        # Generate the beeswarm plot
        plt.figure()
        shap.summary_plot(
            class_shap_values,
            class_samples,
            plot_type="dot",  # Beeswarm plot is the default "dot" plot
            max_display=5,  # Display top 20 features
            show=False
        )
        plt.title(f"SHAP Beeswarm Plot - Class {class_mapping[class_index]}")

        # Save the plot
        output_path = f"{output_dir}/Shap_Beeswarm_Plot_Class_{class_mapping[class_index]}_{model.__class__.__name__}.png"
        plt.savefig(output_path, format="png", bbox_inches="tight", dpi=500)
        plt.close()

    print(f"Class-specific SHAP beeswarm plots for all samples have been saved to {output_dir}.")


def plot_shap_bar_for_all_classes(model, X_test, y_test, class_mapping, output_dir):
    """
    Generate SHAP bar plots for all samples, grouped by class.
    Handles binary and multiclass classification scenarios.
    """
    import shap
    import matplotlib.pyplot as plt
    import os

    shap.initjs()

    # Handle ensemble models
    if hasattr(model, "estimators_") or isinstance(model, (VotingClassifier, StackingClassifier)):
        print(f"Detected ensemble model: {model.__class__.__name__}")
        for idx, sub_model in enumerate(model.estimators_):
            print(f"Processing sub-model {idx + 1}/{len(model.estimators_)}: {sub_model.__class__.__name__}")
            plot_shap_bar_for_all_classes(sub_model, X_test, y_test, class_mapping, output_dir)
        return

    # Determine the appropriate SHAP explainer
    if hasattr(model, 'tree_'):
        explainer = shap.TreeExplainer(model)  # Tree-based models
    elif isinstance(model, (LogisticRegression, SGDClassifier, LinearDiscriminantAnalysis)):
        explainer = shap.LinearExplainer(model, X_test)  # Linear models
    elif isinstance(model, (SVC, GaussianNB, KNeighborsClassifier)):
        if hasattr(model, 'predict_proba'):
            explainer = shap.KernelExplainer(model.predict_proba, shap.kmeans(X_test, min(50, X_test.shape[0])))
        else:
            raise ValueError(f"SVM models without 'probability=True' are not supported.")
    elif isinstance(model, (XGBClassifier, GradientBoostingClassifier, AdaBoostClassifier)):
        explainer = shap.TreeExplainer(model)  # Boosting models
    else:
        explainer = shap.Explainer(model, X_test)  # General models

    # Generate SHAP values
    shap_values = explainer(X_test)

    # Handle binary and multiclass cases
    if isinstance(shap_values.values, list):  # Multiclass case: SHAP returns a list of arrays
        num_classes = len(shap_values.values)
        shap_values_list = shap_values.values
    elif len(shap_values.values.shape) == 2:  # Binary classification or single-output
        num_classes = 2  # Binary case
        shap_values_list = [shap_values.values, -shap_values.values]
    else:  # Multiclass with a single SHAP array
        num_classes = shap_values.values.shape[2]
        shap_values_list = [shap_values[..., class_index].values for class_index in range(num_classes)]

    # Generate bar plots for each class
    for class_index in range(num_classes):
        # Filter samples for the current class
        class_samples_mask = (y_test == class_index)
        if not class_samples_mask.any():
            print(f"No samples found for class {class_mapping[class_index]}. Skipping.")
            continue

        class_samples = X_test[class_samples_mask]
        class_shap_values = shap_values_list[class_index][class_samples_mask]

        # Compute mean absolute SHAP values for features
        mean_shap_values = np.abs(class_shap_values).mean(axis=0)

        # Generate the bar plot
        plt.figure(figsize=(10, 8))
        plt.barh(
            y=X_test.columns,
            width=mean_shap_values,
            color='skyblue'
        )
        plt.title(f"SHAP Bar Plot - Class {class_mapping[class_index]}")
        plt.xlabel("Mean |SHAP Value|")
        plt.ylabel("Features")
        plt.gca().invert_yaxis()

        # Save the plot
        output_path = f"{output_dir}/Shap_Bar_Plot_Class_{class_mapping[class_index]}_{model.__class__.__name__}.png"
        plt.savefig(output_path, format="png", bbox_inches="tight", dpi=500)
        plt.close()

    print(f"Class-specific SHAP bar plots for all samples have been saved to {output_dir}.")


def plot_shap_bar_for_all_classes(model, X_test, y_test, class_mapping, output_dir):
    """
    Generate SHAP bar plots for all samples, grouped by class.
    Handles binary and multiclass classification scenarios.
    """
    import shap
    import matplotlib.pyplot as plt
    import os

    shap.initjs()

    # Handle ensemble models
    if hasattr(model, "estimators_") or isinstance(model, (VotingClassifier, StackingClassifier)):
        print(f"Detected ensemble model: {model.__class__.__name__}")
        for idx, sub_model in enumerate(model.estimators_):
            print(f"Processing sub-model {idx + 1}/{len(model.estimators_)}: {sub_model.__class__.__name__}")
            plot_shap_bar_for_all_classes(sub_model, X_test, y_test, class_mapping, output_dir)
        return

    # Determine the appropriate SHAP explainer
    if hasattr(model, 'tree_'):
        explainer = shap.TreeExplainer(model)  # Tree-based models
    elif isinstance(model, (LogisticRegression, SGDClassifier, LinearDiscriminantAnalysis)):
        explainer = shap.LinearExplainer(model, X_test)  # Linear models
    elif isinstance(model, (SVC, GaussianNB, KNeighborsClassifier)):
        if hasattr(model, 'predict_proba'):
            explainer = shap.KernelExplainer(model.predict_proba, shap.kmeans(X_test, min(50, X_test.shape[0])))
        else:
            raise ValueError(f"SVM models without 'probability=True' are not supported.")
    elif isinstance(model, (XGBClassifier, GradientBoostingClassifier, AdaBoostClassifier)):
        explainer = shap.TreeExplainer(model)  # Boosting models
    else:
        explainer = shap.Explainer(model, X_test)  # General models

    # Generate SHAP values
    shap_values = explainer(X_test)

    # Handle binary and multiclass cases
    if isinstance(shap_values.values, list):  # Multiclass case: SHAP returns a list of arrays
        num_classes = len(shap_values.values)
        shap_values_list = shap_values.values
    elif len(shap_values.values.shape) == 2:  # Binary classification or single-output
        num_classes = 2  # Binary case
        shap_values_list = [shap_values.values, -shap_values.values]
    else:  # Multiclass with a single SHAP array
        num_classes = shap_values.values.shape[2]
        shap_values_list = [shap_values[..., class_index].values for class_index in range(num_classes)]

    # Generate bar plots for each class
    for class_index in range(num_classes):
        # Filter samples for the current class
        class_samples_mask = (y_test == class_index)
        if not class_samples_mask.any():
            print(f"No samples found for class {class_mapping[class_index]}. Skipping.")
            continue

        class_samples = X_test[class_samples_mask]
        class_shap_values = shap_values_list[class_index][class_samples_mask]

        # Compute mean absolute SHAP values for features
        mean_shap_values = np.abs(class_shap_values).mean(axis=0)

        # Generate the bar plot
        plt.figure(figsize=(10, 8))
        plt.barh(
            y=X_test.columns,
            width=mean_shap_values,
            color='skyblue'
        )
        plt.title(f"SHAP Bar Plot - Class {class_mapping[class_index]}")
        plt.xlabel("Mean |SHAP Value|")
        plt.ylabel("Features")
        plt.gca().invert_yaxis()

        # Save the plot
        output_path = f"{output_dir}/Shap_Bar_Plot_Class_{class_mapping[class_index]}_{model.__class__.__name__}.png"
        plt.savefig(output_path, format="png", bbox_inches="tight", dpi=500)
        plt.close()

    print(f"Class-specific SHAP bar plots for all samples have been saved to {output_dir}.")


def plot_shap_bar_multiclass_sorted(model, X_test, y_test, class_mapping, output_dir):
    """
    Generate a SHAP bar plot for all features grouped by class.
    Features are sorted by mean importance, and colors are fixed to red and blue for binary or multiclass.
    """
    import shap
    import matplotlib.pyplot as plt
    import numpy as np

    shap.initjs()

    # Handle ensemble models
    if hasattr(model, "estimators_") or isinstance(model, (VotingClassifier, StackingClassifier)):
        print(f"Detected ensemble model: {model.__class__.__name__}")
        for idx, sub_model in enumerate(model.estimators_):
            print(f"Processing sub-model {idx + 1}/{len(model.estimators_)}: {sub_model.__class__.__name__}")
            plot_shap_bar_multiclass_sorted(sub_model, X_test, y_test, class_mapping, output_dir)
        return

    # Determine the appropriate SHAP explainer
    if hasattr(model, 'tree_'):
        explainer = shap.TreeExplainer(model)  # Tree-based models
    elif isinstance(model, (LogisticRegression, SGDClassifier, LinearDiscriminantAnalysis)):
        explainer = shap.LinearExplainer(model, X_test)  # Linear models
    elif isinstance(model, (SVC, GaussianNB, KNeighborsClassifier)):
        if hasattr(model, 'predict_proba'):
            explainer = shap.KernelExplainer(model.predict_proba, shap.kmeans(X_test, min(50, X_test.shape[0])))
        else:
            raise ValueError(f"SVM models without 'probability=True' are not supported.")
    elif isinstance(model, (XGBClassifier, GradientBoostingClassifier, AdaBoostClassifier)):
        explainer = shap.TreeExplainer(model)  # Boosting models
    else:
        explainer = shap.Explainer(model, X_test)  # General models

    # Generate SHAP values
    shap_values = explainer(X_test)

    # Handle binary and multiclass cases
    if isinstance(shap_values.values, list):  # Multiclass case: SHAP returns a list of arrays
        num_classes = len(shap_values.values)
        shap_values_list = shap_values.values
    elif len(shap_values.values.shape) == 2:  # Binary classification or single-output
        num_classes = 2  # Binary case
        shap_values_list = [shap_values.values, -shap_values.values]
    else:  # Multiclass with a single SHAP array
        num_classes = shap_values.values.shape[2]
        shap_values_list = [shap_values[..., class_index].values for class_index in range(num_classes)]

    # Initialize bar width and colors
    bar_colors = ["blue", "red"] if num_classes == 2 else ["blue", "red", "purple", "green"]  # Extend for multiclass

    # Combine SHAP values across all classes
    feature_names = X_test.columns
    feature_shap_values = np.zeros((len(feature_names), num_classes))

    for class_index in range(num_classes):
        class_samples_mask = (y_test == class_index)
        if not class_samples_mask.any():
            print(f"No samples found for class {class_mapping[class_index]}. Skipping.")
            continue

        class_samples = X_test[class_samples_mask]
        class_shap_values = shap_values_list[class_index][class_samples_mask]

        # Compute mean absolute SHAP values for features
        feature_shap_values[:, class_index] = np.abs(class_shap_values).mean(axis=0)

    # Sort features by total importance
    total_importance = feature_shap_values.sum(axis=1)
    sorted_indices = np.argsort(total_importance)[::-1]  # Descending order
    sorted_feature_names = feature_names[sorted_indices]
    sorted_feature_shap_values = feature_shap_values[sorted_indices, :]

    # Plot bar chart for SHAP values by class
    fig, ax = plt.subplots(figsize=(12, 8))
    for class_index, class_name in class_mapping.items():
        ax.barh(
            sorted_feature_names,
            sorted_feature_shap_values[:, class_index],
            left=np.sum(sorted_feature_shap_values[:, :class_index], axis=1),
            color=bar_colors[class_index % len(bar_colors)],
            edgecolor='black',
            label=class_name,
        )

    ax.set_xlabel('Mean |SHAP Value| (Average impact on model output)', fontsize=12)
    ax.set_title('SHAP Bar Plot - Feature Importance by Class (Sorted)', fontsize=16)
    ax.legend(title='Class', loc='best', fontsize=10)
    ax.invert_yaxis()

    # Save the plot
    output_path = os.path.join(output_dir, f'Shap_Bar_Plot_Multiclass_Sorted_{model.__class__.__name__}.png')
    plt.savefig(output_path, format="png", bbox_inches="tight", dpi=500)
    plt.close()

    print(f"SHAP bar plot for all classes (sorted) has been saved to {output_dir}.")


###########################################
#      LIME Explainability Plot Functions
###########################################
def plot_lime_explanation(model, X_train, X_test, class_mapping, index, num_features, output_dir):
    """
    Generate and save LIME explanation for a specific instance.
    """
    explainer = lime_tabular.LimeTabularExplainer(
        training_data=X_train.values, 
        #feature_names=X_train.columns,
        feature_names=X_train.columns.tolist(),
        class_names=[class_mapping[i] for i in range(len(class_mapping))],  # class_names=['0', '1', '2'],
        discretize_continuous=True,
        mode='classification'
    )

    # Generate explanation
    exp = explainer.explain_instance(
        data_row=X_test.iloc[index].values, 
        predict_fn=model.predict_proba, 
        num_features=num_features
        )    #default: num_features=10
    

    # Save to HTML | pickle file
    exp.save_to_file(f'{output_dir}/lime_explanation_{model.__class__.__name__}.html')
    joblib.dump(exp, f'{output_dir}/lime_explanation_{model.__class__.__name__}.pkl')

    print(f"Explanation saved to {output_dir}/lime_explanation.html and {output_dir}/lime_explanation.pkl")


    # Optionally, adjust plot size and save a high-quality plot as PNG
    fig = exp.as_pyplot_figure()
    fig.set_size_inches(12, 10)  # Adjust figure size for better readability
    plt.subplots_adjust(left=0.3)  # Increase left margin
    fig.savefig(f'{output_dir}/lime_explanation_{model.__class__.__name__}.png', dpi=500)
    plt.close(fig)  # Close the figure to avoid display issues in Jupyter notebooks


#############################################



