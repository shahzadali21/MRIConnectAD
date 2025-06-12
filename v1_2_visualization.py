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

from sklearn.metrics import confusion_matrix



###########################################
#      Visualization Plot Functions
###########################################

def plot_confusion_matrix(models, X_test, y_test, output_dir):
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
    axs = axs.flatten()
    
    for i, (name, model) in enumerate(models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['CN', 'MCI', 'AD-D'], yticklabels=['CN', 'MCI', 'AD-D'], ax=axs[i])
        axs[i].set_title(f'Confusion Matrix - {name}')
        axs[i].set_xlabel('Predicted Labels')
        axs[i].set_ylabel('True Labels')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/Top_4_ConfusionMatrix.png', format='png', bbox_inches='tight', dpi=500)
    plt.close()
    print(f"Confusion matrix plots for top 4 models saved to {output_dir}.")


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


def plot_type1_type2_errors(metrics, model_names, output_dir):
    """
    Generate a line plot for Type I and Type II error rates for all models and save the plot to the output directory.
    """
    # Reset the index to ensure 'Model' becomes a column in the DataFrame
    metrics_filtered = metrics[['Type I Error', 'Type II Error']].copy()
    metrics_filtered['Model'] = model_names

    # Melt the dataframe for seaborn
    melted_metrics = metrics_filtered.melt(id_vars='Model', var_name='Error Type', value_name='Error Rate')

    plt.figure(figsize=(12, 8))
    
    # Create a line plot
    sns.lineplot(x='Model', y='Error Rate', hue='Error Type', style='Error Type', markers=True, data=melted_metrics)

    plt.title('Type I and Type II Error Rates for All Models')
    plt.xlabel('Models')
    plt.ylabel('Error Rate (%)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True)
    plt.legend(loc='upper right')
    
    plt.savefig(f'{output_dir}/type1_type2_error_lineplot.png', format='png', bbox_inches='tight', dpi=500)
    plt.close()
    print(f"Type I and Type II error rates plot saved to {output_dir}.")


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




