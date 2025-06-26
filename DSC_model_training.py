# module: run_ensemble_nested_cv_full.py
# -*- coding: utf-8 -*-
"""
Run complete ensemble model training and nested cross-validation pipeline
using morphometric + CSF (MO) features. Includes saving models, results, and plots.
"""

import os
import logging
from pathlib import Path
from typing import Tuple
from joblib import load

import pandas as pd
import numpy as np
from sklearn.exceptions import ConvergenceWarning

from DSC_model_utils import (
        load_data, save_data,
        get_models_and_params,
        run_nested_evaluations,
        get_ensemble_model_from_top_models,
        evaluate_ensemble_metrics, evaluate_ensemble_with_bootstrap_ci,
        save_model,
        save_results, save_metrics_to_excel
    )

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn.neural_network')


class ADModelTrainer:
    CLASSIFICATION_TYPES = {
        'binary': ['CN_vs_AD', 'CN_vs_MCI', 'MCI_vs_AD'],
        'three_level': ['CN_MCI_AD']
    }

    FEATURE_COMBINATIONS = ['MO', 'MS', 'GT', 'MO_MS', 'MO_MS_GT', 'DG_MO_MS_GT']  # Add other feature sets as needed

    def __init__(self, output_dir: str = 'DSC_NCV', n_jobs: int = -1):
        self.output_dir = Path(output_dir)
        self.n_jobs = n_jobs if n_jobs != -1 else os.cpu_count() - 1
        self.metrics_dict = {}

    def setup_directories(self, feature_combo: str, classification_type: str, comparison: str) -> Tuple[Path, Path, Path, Path]:
        comp_folder = comparison.replace('vs_', '') if classification_type == 'binary' else comparison
        base_dir = self.output_dir / feature_combo / comp_folder
        data_dir = base_dir / 'data'
        models_dir = base_dir / 'models'
        results_dir = base_dir / 'results'
        plots_dir = base_dir / 'plots'

        for d in [data_dir, models_dir, results_dir, plots_dir]:
            d.mkdir(parents=True, exist_ok=True)

        return data_dir, models_dir, results_dir, plots_dir

    def load_and_validate_data0(self, data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        logging.info("Loading preprocessed data")
        X_train = pd.read_csv(data_dir / 'X_train.csv')
        X_test = pd.read_csv(data_dir / 'X_test.csv')
        y_train = pd.read_csv(data_dir / 'y_train.csv').squeeze()
        y_test = pd.read_csv(data_dir / 'y_test.csv').squeeze()

        X_train = pd.DataFrame(X_train)
        X_test = pd.DataFrame(X_test)
        y_train = np.asarray(y_train).ravel()
        y_test = np.asarray(y_test).ravel()

        assert len(X_train) == len(y_train), "Train data/label mismatch"
        assert len(X_test) == len(y_test), "Test data/label mismatch"
        assert X_train.shape[1] == X_test.shape[1], "Feature dimension mismatch"

        return X_train, X_test, y_train, y_test
    
    def load_and_validate_data(self, data_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        logging.info("Loading preprocessed data")
        X_train = pd.read_csv(data_dir / 'X_train.csv').to_numpy(copy=True)
        X_test = pd.read_csv(data_dir / 'X_test.csv').to_numpy(copy=True)
        y_train = pd.read_csv(data_dir / 'y_train.csv').squeeze().to_numpy(copy=True).ravel()
        y_test = pd.read_csv(data_dir / 'y_test.csv').squeeze().to_numpy(copy=True).ravel()

        assert len(X_train) == len(y_train), "Train data/label mismatch"
        assert len(X_test) == len(y_test), "Test data/label mismatch"
        assert X_train.shape[1] == X_test.shape[1], "Feature dimension mismatch"

        return X_train, X_test, y_train, y_test

    def run_model_training(self, X_train, y_train, X_test, y_test, models_dir, results_dir, plots_dir) -> pd.DataFrame:
        #X_full = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)
        X_full = np.concatenate([X_train, X_test], axis=0)
        y_full = np.concatenate((y_train, y_test))

        models = get_models_and_params(seed=42)
        _, summary = run_nested_evaluations(
            models_with_params=models,
            X=X_full,
            y=pd.Series(y_full),
            results_path=results_dir,
            models_path=models_dir,
            n_jobs=self.n_jobs
        )

        predictions = {}
        for name in models:
            try:
                best_model = load(models_dir / f'best_{name}.pkl')
                y_pred = best_model.predict(X_test)
                predictions[name] = y_pred
            except Exception as e:
                logging.warning(f"Prediction failed for {name}: {str(e)}")

        ensemble, selected_models = get_ensemble_model_from_top_models(
            summary_df=summary,
            models_dict=models,
            strategy='topk',
            top_k=3,
            seed=42
        )

        ensemble.fit(X_train, y_train)
        save_model(ensemble, models_dir / 'ensemble.pkl')

        # Save list of models selected for ensemble
        ensemble_models_path = models_dir / 'ensemble_selected_models.csv'
        pd.DataFrame({'Selected_Model': selected_models}).to_csv(ensemble_models_path, index=False)
        logging.info(f"Saved selected ensemble models to {ensemble_models_path}")

        y_pred_ens = ensemble.predict(X_test)
        predictions['Ensemble'] = y_pred_ens
        save_results(y_test, predictions, results_dir / 'predictions.csv')

        y_prob_ens = ensemble.predict_proba(X_test) if hasattr(ensemble, "predict_proba") else None
        #ensemble_summary = evaluate_ensemble_metrics(y_test, y_pred_ens, y_prob_ens)
        ensemble_metrics_df = evaluate_ensemble_with_bootstrap_ci(y_true=y_test, y_pred=y_pred_ens, y_prob=y_prob_ens, n_bootstraps=1000)
        # print(ensemble_metrics_df)
        ensemble_metrics_df.to_csv(results_dir / 'ensemble_metrics.csv', index=False)
        #summary = pd.concat([summary, ensemble_summary], ignore_index=True)

        return summary



    def process_classification_task(self, feature_combo, classification_type, comparison):
        logging.info(f"\n{'#' * 60}")
        logging.info(f"Processing: {feature_combo} | {classification_type} | {comparison}")
        logging.info(f"{'#' * 60}")

        data_dir, models_dir, results_dir, plots_dir = self.setup_directories(
            feature_combo, classification_type, comparison
        )
        X_train, X_test, y_train, y_test = self.load_and_validate_data(data_dir)
        summary = self.run_model_training(X_train, y_train, X_test, y_test, models_dir, results_dir, plots_dir)

        key = f"{feature_combo}_{classification_type}_{comparison}"
        self.metrics_dict[key] = summary
        logging.info(f"Finished: {key}")

    def execute_pipeline(self):
        logging.info("=== Starting Ensemble Training Pipeline ===")
        for feature_combo in self.FEATURE_COMBINATIONS:
            self.metrics_dict = {}  # Reset metrics for each feature set

            for classification_type, comparisons in self.CLASSIFICATION_TYPES.items():
                for comparison in comparisons:
                    self.process_classification_task(feature_combo, classification_type, comparison)

            output_file = self.output_dir / f'metrics_{feature_combo}.xlsx'
            save_metrics_to_excel(self.metrics_dict, output_file)
            logging.info(f"Saved summary metrics to {output_file}")
        logging.info("=== Pipeline completed ===")
        return True


def main():
    logging.info("=== Initializing Alzheimer's ML Pipeline ===")

    output_dir = 'DSC_NCV'
    if not Path(output_dir).exists():
        logging.error(f"Missing preprocessed directory: {output_dir}. Run preprocessing first.")
        return

    trainer = ADModelTrainer(output_dir=output_dir, n_jobs=-1)
    success = trainer.execute_pipeline()

    if success:
        logging.info("✅ All tasks completed successfully.")
    else:
        logging.error("❌ Pipeline failed.")


if __name__ == '__main__':
    main()
