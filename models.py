# model.py

import os
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm

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


def get_models_and_params(seed):
    return{
        # Linear
        'LR': (LogisticRegression(random_state=seed), {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'penalty': ['l1', 'l2', 'elasticnet', 'none'], 'solver': ['liblinear', 'lbfgs', 'saga'], 'max_iter': [100, 500, 1000]}),
        'LR-SGD': (SGDClassifier(loss='log_loss', random_state=seed), {'penalty': ['l2', 'l1', 'elasticnet'], 'alpha': [0.0001, 0.001, 0.01, 0.1], 'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'], 'eta0': [0.001, 0.01, 0.1, 1], 'max_iter': [100, 200, 300, 500, 1000, 1500], 'tol': [1e-2, 1e-3, 1e-4, 1e-5]}),
        'LDA': (LinearDiscriminantAnalysis(), {'solver': ['lsqr', 'eigen'], 'shrinkage': ['auto', None, 'log']}),
       
        # Tree-based
        'DT': (DecisionTreeClassifier(random_state=seed), {'criterion': ['gini', 'entropy'], 'max_depth': [3, 5, 10, 15, 20, None], 'min_samples_split': [2, 3, 5, 7, 10], 'max_features': [0.3, 0.5, 0.7, 1.0]}),
        'RF': (RandomForestClassifier(random_state=seed),{'n_estimators': [5, 10, 30, 50, 70], 'max_depth': [3, 5, 10, 15, 20, None],'min_samples_split': [2, 5, 10, 15], 'min_samples_leaf': [1, 2, 4, 6], 'bootstrap': [True, False]}),
        #'ETC': (ExtraTreesClassifier(random_state=seed),{'n_estimators': [5, 10, 30, 50, 70], 'criterion': ['gini', 'entropy'], 'max_depth': [10, 20, 30, None],'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4], 'max_features': [0.3, 0.5, 0.7, 1.0],'bootstrap': [True, False], 'oob_score': [True]}),
        'AdB': (AdaBoostClassifier(random_state=seed),{'n_estimators': [5, 10, 30, 50, 70], 'learning_rate': [0.01, 0.1, 0.5, 1.0],'algorithm': ['SAMME']}),
        'XGBC': (XGBClassifier(random_state=seed),{'n_estimators': [5, 10, 30, 50, 70], 'learning_rate': [0.01, 0.03, 0.1, 0.3, 0.5], 'max_depth': [3, 5, 7, 10],'min_child_weight': [1, 3, 5], 'subsample': [0.8, 1.0], 'colsample_bytree': [0.8, 1.0],'gamma': [0, 0.1, 0.2], 'reg_alpha': [0, 0.1, 0.5], 'reg_lambda': [1, 1.5, 2]}),
        
        # Kernel
        'SVM': (SVC(random_state=seed, probability=True),{'C': [0.01, 0.1, 1, 10, 100, 500, 1000], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'gamma': ['scale', 'auto']}),   # 'max_iter': [1000, 5000, 10000]
        
        # Distance-based
        'KNN': (KNeighborsClassifier(),{'n_neighbors': [1, 3, 5, 7, 9], 'weights': ['uniform', 'distance'], 'p': [1, 2], 'metric': ['euclidean', 'manhattan', 'minkowski']}),
        
        # Probabilistic
        'NB': (GaussianNB(), {'var_smoothing': np.logspace(0, -9, num=100)}), 
        
        # Neural Networks
        'MLP': (MLPClassifier(random_state=seed),{'learning_rate_init': [0.001, 0.0005, 0.0001], 'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0,], 'learning_rate': ['constant', 'invscaling', 'adaptive'],'activation': ['tanh', 'relu'],'hidden_layer_sizes': [(30,), (50,), (100,), (50, 25), (100, 50), (200, 100)], 'solver': ['lbfgs', 'sgd', 'adam'], 'max_iter': [50, 100, 150, 200, 250, 300, 400, 500], 'early_stopping': [True], 'batch_size': [32, 64, 128], 'momentum': [0.5, 0.9, 0.99]}),
        }


def get_ensemble_model_from_top_models(summary_df, models_dict, strategy='mean', top_k=3, seed=42):
    """
    Create an ensemble model from base classifiers based on selection strategy.

    strategy: 'mean' or 'topk'
    """
    if strategy == 'mean':
        mean_acc = summary_df.groupby('Model')['Accuracy']['mean'].mean()
        selected_models = summary_df[summary_df['Accuracy']['mean'] > mean_acc]['Model'].unique()
    elif strategy == 'topk':
        sorted_models = summary_df.groupby('Model')['Accuracy']['mean'].sort_values(ascending=False)
        selected_models = sorted_models.head(top_k).index.tolist()
    else:
        raise ValueError("strategy must be 'mean' or 'topk'")

    ensemble_estimators = [(name.lower(), models_dict[name][0]) for name in selected_models if name in models_dict]

    if not ensemble_estimators:
        raise ValueError("No models selected for ensemble. Check your selection strategy.")

    return VotingClassifier(estimators=ensemble_estimators, voting='soft')


# Hyperparameter search function
def perform_hyperparameter_search(model, param_grid, X, y, method='grid', cv=5, scoring='accuracy', n_iter=10, seed=42):

    if method == 'grid':
        search = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=cv, n_jobs=-1)
    elif method == 'random':
        search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=n_iter,
                                    scoring=scoring, cv=cv, random_state=seed, n_jobs=-1)
    else:
        raise ValueError("Method must be 'grid' or 'random'")
    
    search.fit(X, y)
    return search.best_estimator_, search.best_params_, search.best_score_
    

def nested_cv_evaluation(model, param_grid, X, y, outer_splits=5, inner_splits=5, scoring='accuracy', search_method='grid', n_iter=10):
    outer_cv = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=42)

    results = []

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]


        best_model, best_params, _ = perform_hyperparameter_search(
            model=model,
            param_grid=param_grid,
            X=X_train,
            y=y_train,
            method=search_method,
            cv=inner_cv,
            scoring=scoring,
            n_iter=n_iter
        )

        y_pred = best_model.predict(X_test)

        result = {
            'Fold': fold,
            'Accuracy': round(accuracy_score(y_test, y_pred) * 100, 2),
            'BACC': round(balanced_accuracy_score(y_test, y_pred) * 100, 2),
            'MCC': round(matthews_corrcoef(y_test, y_pred) * 100, 2),
            'Precision': round(precision_score(y_test, y_pred, average='weighted') * 100, 2),
            'Recall': round(recall_score(y_test, y_pred, average='weighted') * 100, 2),
            'F1-Score': round(f1_score(y_test, y_pred, average='weighted') * 100, 2)
        }

        # Calculate AUC (if applicable)
        if hasattr(best_model, "predict_proba"):
            try:
                if len(np.unique(y)) == 2:
                    y_prob = best_model.predict_proba(X_test)[:, 1]
                    result['AUC'] = round(roc_auc_score(y_test, y_prob) * 100, 2)
                else:
                    y_bin = label_binarize(y_test, classes=np.unique(y))
                    y_prob = best_model.predict_proba(X_test)
                    result['AUC'] = round(roc_auc_score(y_bin, y_prob, average='weighted', multi_class='ovr') * 100, 2)
            except Exception:
                result['AUC'] = None
        else:
            result['AUC'] = None

        results.append(result)

    return pd.DataFrame(results)






def summarize_with_ci(df):
    summary = df.drop(columns=['Fold']).agg(['mean', 'std'])
    ci_95 = df.drop(columns=['Fold']).agg(lambda x: np.percentile(x, [2.5, 97.5])).T
    ci_95.columns = ['CI 2.5%', 'CI 97.5%']
    return pd.concat([summary.T, ci_95], axis=1)


def run_nested_evaluations(models_with_params, X, y, outer_splits=5, inner_splits=3, scoring='accuracy', search_method='grid', n_iter=10):
    all_results = {}
    all_summaries = []
    
    for name, (model, param_grid) in tqdm(models_with_params.items(), desc="Nested CV", unit="model"):
        print(f"Running nested CV for: {name}")
        df = nested_cv_evaluation(model, param_grid, X, y, outer_splits, inner_splits, scoring, search_method, n_iter)
        df.to_csv(f"nested_cv_{name}.csv", index=False)

        summary = summarize_with_ci(df)
        summary.insert(0, 'Model', name)
        all_summaries.append(summary.reset_index())

        all_results[name] = df

        final_summary = pd.concat(all_summaries, axis=0)
        final_summary.to_csv("nested_cv_summary_all_models.csv", index=False)
        print("\nSummary statistics saved to nested_cv_summary_all_models.csv")

    return all_results, final_summary
