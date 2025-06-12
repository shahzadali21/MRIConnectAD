import os
import logging
import glob
import pandas as pd

from v2fs_feature_selection import feature_selection
from v2fs_model_optimizing import train_and_evaluate_models

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Feature combination to use
feature_combination = "Demographic_Microstructural_Morphometric_GT"

# Directories and Configuration
output_dir = "V2fs_ProjectOutput"
summary_file = os.path.join(output_dir, f"classification_metrics_summary.xlsx")

# Define classification types and comparisons
CLASSIFICATION_COMPARISONS = {
    'binary': ['CN_AD', 'CN_MCI', 'MCI_AD'],
    'three_level': ['CN_MCI_AD']
}

# Feature selection methods to test
feature_selection_methods = ["fs_mutual_info", "fs_anova", "fs_rfe", "fs_random_forest", "fs_pca"]

# Iterate over classification types and comparisons
metrics_summary = []

for classification_type, comparisons in CLASSIFICATION_COMPARISONS.items():
    for comparison in comparisons:
        logging.info(f"###############################################################")
        logging.info(f"Processing {classification_type} classification: {comparison}")
        logging.info(f"###############################################################")
        
        # Define directory paths for this comparison
        comparison_dir = os.path.join(output_dir, feature_combination, comparison)
        data_dir = os.path.join(comparison_dir, 'data')
        models_dir = os.path.join(comparison_dir, 'models')
        results_dir = os.path.join(comparison_dir, 'results')
        
        # Load data for the current comparison
        X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
        X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
        y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).values.ravel()
        y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv")).values.ravel()
        
        # Perform feature selection and train models iteratively
        for method in feature_selection_methods:
            logging.info(f"Starting feature selection method: {method}")
            method_dir = os.path.join(comparison_dir, method)
            os.makedirs(method_dir, exist_ok=True)
            
            for k in range(20, X_train.shape[1] + 1, 20):  # Increment by 20 features
                logging.info(f"Feature selection method: {method}, Top {k} features")
                
                # Feature selection
                X_train_selected, selected_features = feature_selection(X_train, y_train, method=method, k=k)
                X_test_selected = X_test.iloc[:, selected_features]
                
                # Create directories for this iteration
                iteration_dir = os.path.join(method_dir, f"top_{k}")
                os.makedirs(iteration_dir, exist_ok=True)
                iteration_models_dir = os.path.join(iteration_dir, "models")
                iteration_results_dir = os.path.join(iteration_dir, "results")
                
                # Save reduced datasets
                data_dir_iteration = os.path.join(iteration_dir, "data")
                os.makedirs(data_dir_iteration, exist_ok=True)
                
                pd.DataFrame(X_train_selected, columns=X_train.columns[selected_features]).to_csv(
                    os.path.join(data_dir_iteration, "X_train.csv"), index=False)
                pd.DataFrame(X_test_selected, columns=X_test.columns[selected_features]).to_csv(
                    os.path.join(data_dir_iteration, "X_test.csv"), index=False)
                
                pd.DataFrame(y_train).to_csv(os.path.join(data_dir_iteration, "y_train.csv"), index=False)
                pd.DataFrame(y_test).to_csv(os.path.join(data_dir_iteration, "y_test.csv"), index=False)
                
                # Train and evaluate models
                metrics_dict = {}
                train_and_evaluate_models(
                    data_dir=data_dir_iteration,
                    models_dir=iteration_models_dir,
                    results_dir=iteration_results_dir,
                    classification_type=classification_type,
                    comparison=comparison,
                    metrics_dict=metrics_dict,
                    feature_selection_method=method,
                    k=k
                )
                
                # Save metrics for this iteration
                metrics_file = os.path.join(iteration_results_dir, f"metrics_top_{k}.csv")
                metrics_df = metrics_dict[f"{classification_type}_{comparison}"]
                metrics_df.to_csv(metrics_file, index=True)
                logging.info(f"Saved metrics for {method}, Top {k} features to {metrics_file}")
                
                # Append metrics to the summary
                metrics_summary.append(metrics_df)

# Generate a summary of all metrics
logging.info("Generating summary of all metrics...")
summary_df = pd.concat(metrics_summary, ignore_index=True)

# Save the summary to an Excel file
summary_df.to_excel(summary_file, index=False)
logging.info(f"Metrics summary saved to {summary_file}")
print(f"Metrics summary saved to {summary_file}")
