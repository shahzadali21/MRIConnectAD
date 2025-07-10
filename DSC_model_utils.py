# module: DCS_models_utils.py
# -*- coding: utf-8 -*-
# Author: Shahzad Ali
"""
Enhanced model training, evaluation, and ensemble creation module.
Includes hyperparameter optimization, nested cross-validation, and model persistence.
"""

import os
import logging
import time
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, Tuple, Union, List

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
        AdaBoostClassifier, RandomForestClassifier, VotingClassifier
    )
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, matthews_corrcoef,
        precision_score, recall_score, f1_score, roc_auc_score
    )
from sklearn.preprocessing import label_binarize
from sklearn.base import BaseEstimator
from sklearn.utils import resample



# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_models_and_params(seed: int=42) -> Dict[str, Tuple[BaseEstimator, Dict]]:
    """Return dictionary of models and their parameter grids with metadata."""
    return{
        # Linear Models
        'LogR': (LogisticRegression(random_state=seed), 
               {
                'C': [0.001, 0.01, 0.1, 1, 10, 100], 
                'penalty': ['l1', 'l2', 'elasticnet'], 
                'solver': ['saga'], 
                'max_iter': [100, 500, 1000],
                'class_weight': [None, 'balanced'],
                'l1_ratio': [0.0, 0.25, 0.5, 0.75, 1.0]  # REQUIRED if 'penalty' is 'elasticnet'
                }
            ),
        
        'LR-SGD': (SGDClassifier(random_state=seed), 
                {
                'loss': ['log_loss', 'modified_huber'], 
                'penalty': ['l2', 'l1', 'elasticnet'],
                'alpha': np.logspace(-6, -1, 6), 
                'learning_rate': ['constant', 'invscaling', 'adaptive'],
                'eta0': [0.001, 0.01, 0.1],
                'max_iter': [500, 1000], 
                'tol': [1e-3, 1e-4], 
                'early_stopping': [True],
                'validation_fraction': [0.1],
                'n_iter_no_change': [5],
                'class_weight': ['balanced']
                }
            ),
        
        'LDA': (LinearDiscriminantAnalysis(),
                {
                'solver': ['lsqr'], 
                'shrinkage': ['auto', 0.0, 0.25, 0.5] + list(np.linspace(0, 1, 5)),
                'tol': [1e-4, 1e-3, 1e-2]
                }
            ),
        
        # Tree-based Models
        'DT': (DecisionTreeClassifier(random_state=seed), 
                {
                'criterion': ['gini', 'entropy'], 
                'max_depth': [3, 5, 10, 15, 20, None], 
                'min_samples_split': [2, 5, 10], 
                'min_samples_leaf': [1, 2, 4], 
                'max_features': [0.3, 0.5, 0.7, 1.0],
                'class_weight': [None, 'balanced']
                }
            ),
        
        'RF': (RandomForestClassifier(random_state=seed),
               {
                'n_estimators': [50, 100, 200], 
                'max_depth': [5, 10, 20, None], 
                'min_samples_split': [2, 5], 
                'min_samples_leaf': [1, 2], 
                'bootstrap': [True, False], 
                'max_features': ['sqrt', 'log2'],
                'class_weight': [None, 'balanced']
                }
            ),
        
        'AdaB': (AdaBoostClassifier(random_state=seed),
                {
                'n_estimators': [50, 100, 200], 
                'learning_rate': [0.01, 0.1, 0.5, 1.0], 
                'algorithm': ['SAMME'], 
                'estimator': [
                            None, 
                            DecisionTreeClassifier(max_depth=1), 
                             DecisionTreeClassifier(max_depth=2)]
                }
            ),
        
        'XGB': (XGBClassifier(random_state=seed, eval_metric='logloss'), 
                {
                'n_estimators': [50, 100, 200], 
                'learning_rate': [0.01, 0.1, 0.3],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0],
                'gamma': [0, 0.1], 
                'reg_alpha': [0, 0.1], 
                'reg_lambda': [1, 1.5],
                'scale_pos_weight': [1, 2, 5]  # Imbalance-aware
                }
            ),

        # Kernel
        'SVM': (SVC(random_state=seed, probability=True), 
                {
                'C': [0.1, 1, 10, 100], 
                'kernel': ['linear', 'poly', 'rbf'],
                'degree': [2, 3], 
                'gamma': ['scale', 'auto'],
                'class_weight': [None, 'balanced']
                }
            ),

        # Distance-based
        'KNN': (KNeighborsClassifier(), 
                {
                'n_neighbors': list(range(1, 21, 2)), 
                'weights': ['uniform', 'distance'], 
                'p': [1, 2], 
                'metric': ['euclidean', 'manhattan', 'minkowski']
                }
            ),
        
        # Probabilistic Models
        'NB': (GaussianNB(), 
               {
                'var_smoothing': np.logspace(-12, -3, 20), 
                'priors': [None]
                }
            ),

        # Neural Networks
        'MLP': (MLPClassifier(random_state=seed), 
                {
                'hidden_layer_sizes': [(50,), (100,), (50, 25), (100, 50), (50, 50, 25)], 
                'activation': ['relu', 'tanh'], 
                'solver': ['adam', 'sgd'], 
                'alpha': np.logspace(-5, -1, 5), 
                'learning_rate': ['constant', 'adaptive'], 
                'learning_rate_init': [0.001, 0.0001], 
                'batch_size': [4, 8, 16], 
                'max_iter': [200, 400, 600], 
                'early_stopping': [True], 
                'validation_fraction': [0.1], 
                'beta_1': [0.9, 0.99], 
                'beta_2': [0.999, 0.9999], 
                'epsilon': [1e-8, 1e-5], 
                'n_iter_no_change': [10]
                }
            )
    }


def get_ensemble_model_from_top_models(
    summary_df: pd.DataFrame,
    models_dict: Dict,
    strategy: str = 'mean',
    top_k: int = 3,
    seed: int = 42,
    models_path: str = None
) -> Tuple[VotingClassifier, List[str]]:
    from joblib import load
    import os
    from sklearn.ensemble import VotingClassifier

    # Filter accuracy rows
    accuracy_df = summary_df[summary_df['index'] == 'Accuracy']

    if strategy == 'mean':
        global_mean_acc = accuracy_df['mean'].mean()
        selected_models = accuracy_df[accuracy_df['mean'] > global_mean_acc]['Model'].unique()
        weights = None
    elif strategy == 'topk':
        sorted_df = accuracy_df.sort_values(by=['mean', 'std'], ascending=[False, True])
        selected_models = sorted_df.head(top_k)['Model'].tolist()
        weights = None
    elif strategy == 'weighted':
        sorted_df = accuracy_df.sort_values(by=['mean', 'std'], ascending=[False, True])
        top_models_df = sorted_df.head(top_k)
        selected_models = top_models_df['Model'].tolist()
        weights = top_models_df['mean'].values
    else:
        raise ValueError("strategy must be 'mean', 'topk', or 'weighted'")

    # Load pretrained models from disk
    ensemble_estimators = []
    for name in selected_models:
        model_filename = os.path.join(models_path, f"{name}.pkl") if models_path else None
        if model_filename and os.path.exists(model_filename):
            clf = load(model_filename)
            ensemble_estimators.append((name.lower(), clf))
        else:
            raise FileNotFoundError(f"Trained model for {name} not found at {model_filename}")

    if not ensemble_estimators:
        raise ValueError("No valid models loaded for ensemble creation.")

    voting_model = VotingClassifier(
        estimators=ensemble_estimators,
        voting='soft',
        weights=weights,
        n_jobs=-1
    )

    return voting_model, selected_models



# Hyperparameter search function
def perform_hyperparameter_search(
    model: BaseEstimator,
    param_grid: Dict,
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    method: str = 'grid',
    cv: int = 5,
    scoring: str = 'accuracy',
    n_iter: int = 10,
    n_jobs: int = -1,
    seed: int = 42
) -> Tuple[BaseEstimator, Dict, float]:
    """
    Perform hyperparameter search with validation and timing.
    
    Args:
        model: Scikit-learn estimator
        param_grid: Dictionary of parameters to search
        X: Feature matrix
        y: Target vector
        method: 'grid' or 'random'
        cv: Number of cross-validation folds
        scoring: Evaluation metric
        n_iter: Iterations for random search
        n_jobs: Number of parallel jobs (-1 for all cores)
        seed: Random seed
    
    Returns:
        Tuple of (best_model, best_params, best_score)
    """
    if not isinstance(X, (pd.DataFrame, np.ndarray)):
        raise TypeError("X must be pandas DataFrame or numpy array")
    if not isinstance(y, (pd.Series, np.ndarray)):
        raise TypeError("y must be pandas Series or numpy array")
    
    if n_jobs == -1:
        n_jobs = os.cpu_count() - 1 if os.cpu_count() else 1

    start_time = time.time()
    
    if method == 'grid':
        search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
            verbose=1
        )
    elif method == 'random':
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring=scoring,
            cv=cv,
            random_state=seed,
            n_jobs=n_jobs,
            verbose=1
        )
    else:
        raise ValueError("Method must be 'grid' or 'random'")
    
    search.fit(X, y)
    
    logger.info(f"Hyperparameter search completed in {time.time()-start_time:.2f}s")
    logger.info(f"Best params: {search.best_params_}")
    logger.info(f"Best {scoring}: {search.best_score_:.4f}")
    
    return search.best_estimator_, search.best_params_, search.best_score_
    

# Nested Cross-Validation Evaluation Function
def nested_cv_evaluation(
    model: BaseEstimator,
    param_grid: Dict,
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    outer_splits: int = 5,
    inner_splits: int = 5,
    scoring: str = 'accuracy',
    search_method: str = 'grid',
    n_iter: int = 10,
    n_jobs: int = -1
) -> pd.DataFrame:
    """
    Perform nested cross-validation with timing and feature importance tracking.
    
    Args:
        model: Scikit-learn estimator
        param_grid: Parameter grid for search
        X: Feature matrix
        y: Target vector
        outer_splits: Number of outer CV folds
        inner_splits: Number of inner CV folds
        scoring: Evaluation metric
        search_method: 'grid' or 'random'
        n_iter: Iterations for random search
        n_jobs: Number of parallel jobs
    
    Returns:
        DataFrame with evaluation metrics for each fold, and the best final model
    """
    outer_cv = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=42)

    results = []
    feature_importances = []
    best_model_final = None

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), 1):
        start_time = time.time()
        
        #X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        #y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        best_model, best_params, _ = perform_hyperparameter_search(
            model=model,
            param_grid=param_grid,
            X=X_train,
            y=y_train,
            method=search_method,
            cv=inner_cv,
            scoring=scoring,
            n_iter=n_iter,
            n_jobs=n_jobs
        )

        # Training and prediction
        y_pred = best_model.predict(X_test)
        train_time = time.time() - start_time

        # Store feature importance if available
        if hasattr(best_model, 'feature_importances_'):
            feature_importances.append(best_model.feature_importances_)
        elif hasattr(best_model, 'coef_'):
            feature_importances.append(best_model.coef_)

        # Calculate metrics
        result = {
            'Fold': fold,
            'Accuracy': round(accuracy_score(y_test, y_pred) * 100, 2),
            'BACC': round(balanced_accuracy_score(y_test, y_pred) * 100, 2),
            'MCC': round(matthews_corrcoef(y_test, y_pred) * 100, 2),
            'Precision': round(precision_score(y_test, y_pred, average='weighted') * 100, 2),
            'Recall': round(recall_score(y_test, y_pred, average='weighted') * 100, 2),
            'F1-Score': round(f1_score(y_test, y_pred, average='weighted') * 100, 2),
            'Training_Time': round(train_time, 2)
        }

        # Calculate AUC if applicable
        if hasattr(best_model, "predict_proba"):
            try:
                if len(np.unique(y)) == 2:
                    y_prob = best_model.predict_proba(X_test)[:, 1]
                    result['AUC'] = round(roc_auc_score(y_test, y_prob) * 100, 2)
                else:
                    y_bin = label_binarize(y_test, classes=np.unique(y))
                    y_prob = best_model.predict_proba(X_test)
                    result['AUC'] = round(roc_auc_score(y_bin, y_prob, average='weighted', multi_class='ovr') * 100, 2)
            except Exception as e:
                logger.warning(f"AUC calculation failed: {str(e)}")
                result['AUC'] = None
        else:
            result['AUC'] = None

        results.append(result)
        best_model_final = best_model  # Last best model used for saving

    # Add feature importance summary if available
    if feature_importances:
        try:
            mean_importance = np.array(feature_importances).mean(axis=0).flatten()
            if len(mean_importance) == len(X.columns):
                importance_df = pd.DataFrame({
                    'Feature': X.columns,
                    'Mean_Importance': mean_importance
                }).sort_values('Mean_Importance', ascending=False)
                logger.info("\nFeature Importance Summary:\n" + importance_df.to_string())
        except Exception as e:
            logger.warning(f"Feature importance extraction failed: {e}")

    return pd.DataFrame(results), best_model_final


# Summarize results with confidence intervals
def summarize_with_ci(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate summary statistics with confidence intervals.
    
    Args:
        df: DataFrame with evaluation metrics
    
    Returns:
        DataFrame with mean, std, and confidence intervals
    """
    summary = df.drop(columns=['Fold']).agg(['mean', 'std'])
    ci_95 = df.drop(columns=['Fold']).agg(lambda x: np.percentile(x, [2.5, 97.5])).T
    ci_95.columns = ['CI 2.5%', 'CI 97.5%']
    return pd.concat([summary.T, ci_95], axis=1)


# Nested Cross-Validation Evaluation Runner
def run_nested_evaluations(
    models_with_params: Dict,
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    outer_splits: int = 5,
    inner_splits: int = 3,
    scoring: str = 'accuracy',
    search_method: str = 'grid',
    n_iter: int = 10,
    n_jobs: int = -1,
    results_path: str = None,
    models_path: str = None,

) -> Tuple[Dict, pd.DataFrame]:
    """
    Run nested CV for multiple models and save/load best cross-validated models.
    If model is already saved, it will skip training and use cached results.
    """
    all_results = {}
    all_summaries = []
    
    if results_path:
        os.makedirs(results_path, exist_ok=True)
    if models_path:
        os.makedirs(models_path, exist_ok=True)

    for name, model_info in tqdm(models_with_params.items(), desc="Nested CV", unit="model"):
        logger.info(f"\nRunning nested CV for: {name}")
        model_path = os.path.join(models_path, f"{name}.pkl")
        result_path = os.path.join(results_path, f"nested_cv_{name}.csv")
        skip_model = os.path.exists(model_path) and os.path.exists(result_path)

        #logger.info(f"Model: {model_info[0].__class__.__name__}, Params: {len(model_info[1])} hyperparams")
        
        if skip_model:
            logger.info(f"✅ Skipping {name}: already optimized and saved.")
            best_model = load_model(model_path)
            df = pd.read_csv(result_path)
        else:
            df, best_model = nested_cv_evaluation(
                model=model_info[0],
                param_grid=model_info[1],
                X=X,
                y=y,
                outer_splits=outer_splits,
                inner_splits=inner_splits,
                scoring=scoring,
                search_method=search_method,
                n_iter=n_iter,
                n_jobs=n_jobs
            )

            # Save newly trained model/results
            joblib.dump(best_model, model_path)
            df.to_csv(result_path, index=False)
            logger.info(f"✅ Saved optimized model to: {model_path}")
            logger.info(f"✅ Saved evaluation results to: {result_path}")

        # Summarize and store
        summary = summarize_with_ci(df)
        summary.insert(0, 'Model', name)
        all_summaries.append(summary.reset_index())
        all_results[name] = df

    # Combine and save final summary
    final_summary = pd.concat(all_summaries, axis=0)

    
    if results_path:
        summary_path = os.path.join(results_path, "NestedCV_summary_all_models.csv")
        final_summary.to_csv(summary_path, index=False)
        logger.info(f"\n📊 Summary statistics saved to: {summary_path}")

    return all_results, final_summary




def evaluate_ensemble_metrics(y_test, y_pred, y_prob) -> pd.DataFrame:
    """
    Compute evaluation metrics for ensemble predictions and return a summarized DataFrame.
    """
    try:
        if y_prob is not None and len(np.unique(y_test)) == 2:
            auc = roc_auc_score(y_test, y_prob[:, 1])
        elif y_prob is not None:
            y_bin = label_binarize(y_test, classes=np.unique(y_test))
            auc = roc_auc_score(y_bin, y_prob, average='weighted', multi_class='ovr')
        else:
            auc = None
    except Exception as e:
        logging.warning(f"AUC computation failed: {e}")
        auc = None

    result = pd.DataFrame([{
        'Fold': 0,
        'Accuracy': round(accuracy_score(y_test, y_pred) * 100, 2),
        'BACC': round(balanced_accuracy_score(y_test, y_pred) * 100, 2),
        'MCC': round(matthews_corrcoef(y_test, y_pred) * 100, 2),
        'Precision': round(precision_score(y_test, y_pred, average='weighted', zero_division=0) * 100, 2),
        'Recall': round(recall_score(y_test, y_pred, average='weighted', zero_division=0) * 100, 2),
        'F1-Score': round(f1_score(y_test, y_pred, average='weighted', zero_division=0) * 100, 2),
        'AUC': auc,
        'Training_Time': 0
    }])

    summary = summarize_with_ci(result)
    summary.insert(0, 'Model', 'Ensemble')
    return summary.reset_index()


def evaluate_ensemble_with_bootstrap_ci(y_true, y_pred, y_prob=None, n_bootstraps=1000, random_state=42):
    np.random.seed(random_state)
    metrics = {
        'Accuracy': [],
        'BACC': [],
        'MCC': [],
        'Precision': [],
        'Recall': [],
        'F1-Score': [],
        'AUC': []
    }

    for i in range(n_bootstraps):
        # Bootstrap sample
        indices = resample(np.arange(len(y_true)), replace=True, random_state=random_state + i)
        y_t = y_true[indices]
        y_p = y_pred[indices]
        y_pr = y_prob[indices] if y_prob is not None else None

        metrics['Accuracy'].append(accuracy_score(y_t, y_p))
        metrics['BACC'].append(balanced_accuracy_score(y_t, y_p))
        metrics['MCC'].append(matthews_corrcoef(y_t, y_p))
        metrics['Precision'].append(precision_score(y_t, y_p, average='weighted', zero_division=0))
        metrics['Recall'].append(recall_score(y_t, y_p, average='weighted', zero_division=0))
        metrics['F1-Score'].append(f1_score(y_t, y_p, average='weighted', zero_division=0))

        if y_prob is not None:
            try:
                if len(np.unique(y_true)) == 2:
                    metrics['AUC'].append(roc_auc_score(y_t, y_pr[:, 1]))
                else:
                    y_bin = label_binarize(y_t, classes=np.unique(y_true))
                    metrics['AUC'].append(roc_auc_score(y_bin, y_pr, average='weighted', multi_class='ovr'))
            except Exception as e:
                logging.warning(f"AUC bootstrap failed: {e}")
                metrics['AUC'].append(np.nan)
        else:
            metrics['AUC'].append(np.nan)

    # Prepare summary
    summary = {}
    for metric, values in metrics.items():
        values = np.array(values)
        summary[metric] = {
            'mean': round(np.nanmean(values) * 100, 2),
            'std': round(np.nanstd(values) * 100, 2),
            'ci_lower': round(np.nanpercentile(values, 2.5) * 100, 2),
            'ci_upper': round(np.nanpercentile(values, 97.5) * 100, 2)
        }

    df = pd.DataFrame(summary).T
    df.columns = ['mean', 'std', 'CI 2.5', 'CI 97.5']
    df.index.name = 'Metric'
    return df.reset_index()



# Save and Load Model Functions
def save_model(model: BaseEstimator, path: str) -> None:
    """Save trained model to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    logger.info(f"Model saved to {path}")

def load_model(path: str) -> BaseEstimator:
    """Load trained model from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"No model found at {path}")
    return joblib.load(path)


def load_data(file_path):
    clinical_df = pd.read_csv(file_path, index_col=False)
    return clinical_df

def save_data(data, file_path):
    """Saves the given DataFrame to the specified file path."""
    data.to_csv(file_path, index=False)
    logging.info(f"Data saved to {file_path}")


def save_results(y_test, predictions_dict, output_file):
    """
    Save the actual and predicted results to a CSV file.
    
    Parameters:
    y_test : array-like : The actual target values.
    predictions_dict : dict : Dictionary of model names and their predictions.
    output_file : str : The path where the results will be saved.
    """
    results_df = pd.DataFrame({'Actual': y_test})  # Ensure y_test is a 1D array
    for model_name, predictions in predictions_dict.items():
        results_df[f'Pred_{model_name}'] = predictions

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    results_df.to_csv(output_file, index=False)
    logging.info(f"Evaluation results/predictions saved to {output_file}")

    
def save_metrics(metrics, output_file):
    """ Save the evaluation metrics to a CSV file with a header for the model names. """
    # Ensure the index (model names) has a name
    metrics.index.name = 'Model'
    
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save the metrics DataFrame to a CSV file
    metrics.to_csv(output_file, index=True)
    
    print(f"Metrics saved to {output_file}")

def save_metrics_to_excel_v0(metrics_dict, output_file):
    """ Save all evaluation metrics to a single Excel file with multiple sheets. """
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        for classification_type, metrics in metrics_dict.items():
            metrics.to_excel(writer, sheet_name=classification_type, index_label='Model')
            logging.info(f"Metrics saved to {output_file} in sheet: {classification_type}")

def save_metrics_to_excel(metrics_dict, output_file):
    """ Save all evaluation metrics to an Excel file with multiple sheets, updating existing sheets if necessary. """
    try:
        # Check if the file exists to avoid overwriting existing data
        if os.path.exists(output_file):
            with pd.ExcelWriter(output_file, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                for classification_type, metrics in metrics_dict.items():
                    metrics.to_excel(writer, sheet_name=classification_type, index_label='Model')
                    logging.info(f"Metrics updated in {output_file} in sheet: {classification_type}")
        else:
            # If the file doesn't exist, create a new one
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                for classification_type, metrics in metrics_dict.items():
                    metrics.to_excel(writer, sheet_name=classification_type, index_label='Model')
                    logging.info(f"Metrics saved to new {output_file} in sheet: {classification_type}")
    except Exception as e:
        logging.error(f"Error saving metrics to Excel: {e}")