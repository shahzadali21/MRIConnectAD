# Alzheimer's Disease (AD) Classification - Machine Learning Pipeline

## Project Overview
This project implements a machine learning pipeline for the classification of Alzheimer's Disease (AD) into different diagnostic categories: Cognitively Normal (CN), Mild Cognitive Impairment (MCI), and Dementia. The pipeline supports both binary classification (CN vs. Dementia) and three-level classification (CN vs. MCI vs. Dementia) using various machine learning models. It includes preprocessing of multimodal data (clinical, TCK metrics, and graph theory metrics), model training, optimization, voting, explainability, and final model comparison.




## Project Structure
- **preprocessing.py**: Handles the preprocessing of clinical data, including feature scaling and train-test splitting.
- **model_training.py**: Trains various machine learning models and evaluates their performance on the test set.
- **model_optimization.py**: Selects top-performing models and optimizes their hyperparameters using GridSearchCV.
- **voting.py**: Implements voting classifiers using the top-performing models from both the initial and optimized models.
- **final_comparison.py**: Compares the performance of all models (initial, optimized, and voting classifiers) and generates visualizations.
- **main.py**: Orchestrates the entire pipeline, running all steps in sequence.

## Directory Structure
Upon execution, the project will create and organize the following directory structure:
- **data/**: Contains preprocessed training and test datasets.
- **models/**: Stores trained and optimized machine learning models.
- **results/**: Contains evaluation metrics and predictions.
- **plots/**: Stores generated comparison plots and confusion matrices.

ProjectOutput_<Feature_Combination>/
├── CN_vs_AD/               # Classification comparison folder (e.g., CN vs AD)
│   ├── data/               # Folder to store training and test datasets (X_train, X_test, y_train, y_test)
│   ├── models/             # Folder to save trained models
│   ├── plots/              # Folder to save generated plots
│   └── results/            # Folder to save final results
├── CN_vs_MCI/              # Another comparison (e.g., CN vs MCI)
│   ├── data/
│   ├── models/
│   ├── plots/
│   └── results/
└── CN_MCI_AD/              # Three-level classification folder
    ├── data/
    ├── models/
    ├── plots/
    └── results/


## How to Run the Project
### Clone the repository
```
git clone <https://github.com/shahzadali21/AD_ML_pipeline.git>
cd <repository-directory>
```
### Install dependencies
- Python 3.7+
- Ensure you have all the required Python packages installed. You can install them using the following command:
```
pip install -r requirements.txt
```

## Running the Project
### Option 1: Running the Project Step-by-Step
#### Step 1: Data Preprocessing
##### For Clinical Data:
```
python preprocessing.py 
```
##### For Multimodal Data (Clinical + Graph Theory Metrics + TCK Metrics)
```
python preprocessing_multimod.py 
```
#### Step 2: Model Training
```
python model_training.py 
```
#### Step 3: Final Model Comparison
```
python final_comparison.py
```
#### Step 4: Model Optimization
```
python model_optimization.py 
```
#### Step 5: Voting Classifiers
```
python voting.py 
```
#### Step Final Comparison:
```
python final_comparison.py 
```

## Option 2: Full Pipeline Execution
Alternatively, you can execute the entire pipeline from preprocessing to final comparison in one go using the main.py script:
```
python main.py
```


## Contributing
Feel free to submit issues or pull requests if you have any improvements or bug fixes.