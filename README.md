# Alzheimer's Disease (AD) Classification - Machine Learning Pipeline

## Project Overview
This project implements a modular machine learning pipeline for the classification of Alzheimer's Disease (AD) into different disease stage categories: Cognitively Normal (CN), Mild Cognitive Impairment (MCI), and Dementia (ADD). It includes preprocessing of multimodal data (clinical scores, MO and MS features, and graph theory metrics), model training, optimization, ensemble learning, and explainability of ML model's predictions.




## Project Structure
- **preprocessing.py**: Handles the preprocessing of clinical data, including feature scaling and train-test splitting.
- **model_utils.py**: Contains core logic for model definition, optimization, ensemble construction, and metrics evaluation.
- **model_training.py**: Executes model training, nested cross-validation, ensemble learning, and evaluation.
- **explainability.py**: Compares the performance of all models and provide model explainability using SHAP and LIME visualizations.

## Directory Structure
Upon execution, the project will create and organize the following directory structure:
- **data/**: Contains preprocessed training and test datasets.
- **models/**: Stores trained and optimized machine learning models.
- **results/**: Contains evaluation metrics and predictions.
- **plots/**: Stores generated comparison plots and confusion matrices.

DSC_NCV/
├── MO/                         # Morphometric + CSF features
│   ├── CN_AD/                  # CN vs AD binary classification
│   │   ├── data/               # Train/test datasets (X_train.csv, X_test.csv, etc.)
│   │   ├── models/             # Trained models and ensemble.pkl
│   │   ├── plots/              # Confusion matrices, ROC curves
│   │   └── results/            # Metrics, predictions.csv, summary files
│   ├── CN_MCI/
│   ├── MCI_AD/
│   └── CN_MCI_AD/              # Three-class classification
├── MS/                         # Microstructural + Structural features
│   └── ...
├── GT/                         # Graph Theory metrics
│   └── ...
├── metrics_MO.xlsx             # Summary Excel file for MO feature metrics
├── metrics_MS.xlsx             # Summary Excel file for MS feature metrics
└── metrics_GT.xlsx             # Summary Excel file for GT feature metrics


## Contributing
Feel free to open issues or submit pull requests if you have improvements or suggestions. Contributions are welcome!