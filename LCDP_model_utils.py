# module: LCDP_models_utils.py
# -*- coding: utf-8 -*-
# Author: Shahzad Ali


import os
import time
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
from typing import Tuple, Dict, Union
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, VotingRegressor, BaggingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, DotProduct
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.utils import resample
from math import log

import warnings
warnings.filterwarnings('ignore')


# --- Data IO ---
def load_data(file_path):
    return pd.read_csv(file_path, index_col=False)

def save_data(data, file_path):
    data.to_csv(file_path, index=False)
    logging.info(f"Data saved to {file_path}")

def save_model(model, name: str, models_dir: str):
    path = os.path.join(models_dir, f"{name}.pkl")
    joblib.dump(model, path)
    logging.info(f"Saved model to: {path}")

def load_model(name: str, models_dir: str):
    path = os.path.join(models_dir, f"{name}.pkl")
    return joblib.load(path)



# --- Model Definitions ---
def get_models_and_params(seed):
    return {
        'LR': (LinearRegression(), {
            'fit_intercept': [True, False]
        }),
        'Ridge': (Ridge(random_state=seed), {
            'alpha': [0.001, 0.01, 0.1, 1, 10, 100]
        }),
        'Lasso': (Lasso(random_state=seed), {
            'alpha': [0.001, 0.01, 0.1, 1, 10, 100]
        }),
        'ElasticNet': (ElasticNet(random_state=seed), {
            'alpha': [0.001, 0.01, 0.1, 1],
            'l1_ratio': [0.1, 0.5, 0.7, 1.0]
        }),
        'DT': (DecisionTreeRegressor(random_state=seed), {
            'max_depth': [3, 5, 10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'max_features': [0.3, 0.5, 0.7, 1.0]
        }),
        'RF': (RandomForestRegressor(random_state=seed), {
            'n_estimators': [10, 50, 100],
            'max_depth': [3, 5, 10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        }),
        'AdaB': (AdaBoostRegressor(random_state=seed), {
            'n_estimators': [10, 50, 100],
            'learning_rate': [0.01, 0.1, 0.5, 1.0]
        }),
        'XGB': (XGBRegressor(random_state=seed), {
            'n_estimators': [10, 50, 100],
            'learning_rate': [0.01, 0.1, 0.3],
            'max_depth': [3, 5, 7],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
            'gamma': [0, 0.1, 0.2],
            'reg_alpha': [0, 0.1, 0.5],
            'reg_lambda': [1, 1.5, 2]
        }),
        'Bagging': (BaggingRegressor(random_state=seed), {
            'n_estimators': [10, 50, 100],
            'max_samples': [0.3, 0.5, 0.7, 1.0],
            'max_features': [0.3, 0.5, 0.7, 1.0]
        }),
        'SVR': (SVR(), {
            'C': [0.01, 0.1, 1, 10, 100],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'epsilon': [0.01, 0.1, 1]
        }),
        'GPR': (GaussianProcessRegressor(), {
            'kernel': [
                1.0 * RBF(length_scale=1.0),
                1.0 * Matern(length_scale=1.0, nu=1.5),
                1.0 * RationalQuadratic(length_scale=1.0, alpha=1.0),
                1.0 * DotProduct(sigma_0=1.0)
            ],
            'alpha': [1e-10, 1e-5, 1e-2],
            'n_restarts_optimizer': [0, 5, 10]
        }),
        'MLP': (MLPRegressor(random_state=seed), {
            'hidden_layer_sizes': [(50,), (100,), (50, 25), (100, 50), (50, 50, 25)],
            'activation': ['relu', 'tanh'],
            'solver': ['adam'],
            'alpha': [0.0001, 0.001],
            'learning_rate': ['constant', 'adaptive'],
            'learning_rate_init': [0.001, 0.0001],
            'max_iter': [500]
        })
    }



# --- Nested CV ---
def nested_cv_regression(
    model, param_grid, X, y,
    outer_splits=5, inner_splits=3,
    scoring='r2', n_jobs=-1
) -> Tuple[pd.DataFrame, object]:
    outer_cv = KFold(n_splits=outer_splits, shuffle=True, random_state=42)
    inner_cv = KFold(n_splits=inner_splits, shuffle=True, random_state=42)

    results = []
    best_model_final = None

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring=scoring,
            cv=inner_cv,
            n_jobs=n_jobs
        )
        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        y_pred = best_model.predict(X_test)

        rss = np.sum((y_test - y_pred) ** 2)
        n = len(y_test)
        p = X_test.shape[1] + 1
        sigma_squared = rss / n

        metrics = {
            'Fold': fold,
            'MAE': mean_absolute_error(y_test, y_pred),
            'MSE': mean_squared_error(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'R2': r2_score(y_test, y_pred),
            'AIC': n * np.log(rss / n) + 2 * p,
            'BIC': n * np.log(rss / n) + p * np.log(n)
        }

        results.append(metrics)
        best_model_final = best_model

    return pd.DataFrame(results), best_model_final


# --- Ensemble Creation ---
def build_voting_regressor(selected_models: Dict[str, object]) -> VotingRegressor:
    estimators = [(name, model) for name, model in selected_models.items()]
    return VotingRegressor(estimators=estimators, n_jobs=-1)


# --- Ensemble Evaluation with Bootstrapping ---
def evaluate_regression_ensemble_bootstrap(y_true, y_pred, n_bootstraps=1000, random_state=42):
    np.random.seed(random_state)
    metrics = {
        'MAE': [], 'MSE': [], 'RMSE': [], 'R2': [], 'AIC': [], 'BIC': []
    }

    n = len(y_true)
    p = 1  # placeholder for model complexity, will be overwritten dynamically

    for i in range(n_bootstraps):
        indices = resample(np.arange(n), replace=True, random_state=random_state + i)
        y_t, y_p = y_true[indices], y_pred[indices]

        rss = np.sum((y_t - y_p) ** 2)
        p = 2  # assumed
        metrics['MAE'].append(mean_absolute_error(y_t, y_p))
        metrics['MSE'].append(mean_squared_error(y_t, y_p))
        metrics['RMSE'].append(np.sqrt(mean_squared_error(y_t, y_p)))
        metrics['R2'].append(r2_score(y_t, y_p))
        metrics['AIC'].append(n * np.log(rss / n) + 2 * p)
        metrics['BIC'].append(n * np.log(rss / n) + p * np.log(n))

    summary = {
        metric: {
            'mean': np.mean(vals),
            'std': np.std(vals),
            'ci_lower': np.percentile(vals, 2.5),
            'ci_upper': np.percentile(vals, 97.5)
        } for metric, vals in metrics.items()
    }

    df = pd.DataFrame(summary).T
    df.columns = ['Mean', 'Std', 'CI 2.5%', 'CI 97.5%']
    df.index.name = 'Metric'
    return df.reset_index()


# --- Model Evaluation ---
def evaluate_models(models, X_test, y_test):
    eval_metrics = []
    predictions = {}
    n = len(y_test)

    for name, model in models.items():
        y_pred = model.predict(X_test)
        predictions[name] = y_pred

        rss = np.sum((y_test - y_pred) ** 2)
        p = X_test.shape[1] + 1

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        aic = n * np.log(rss / n) + 2 * p
        bic = n * np.log(rss / n) + p * np.log(n)

        eval_metrics.append({
            'Model': name,
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2,
            'AIC': aic,
            'BIC': bic
        })

    return pd.DataFrame(eval_metrics), predictions





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

