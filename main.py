"""
This is the main script for orchestrating the entire machine learning pipeline.
It parses command-line arguments, coordinates the execution of preprocessing,
model training, explainability, optimization, voting, and final comparison steps,
and manages output directories for storing results.
"""

import os
import logging
import argparse
import subprocess

os.environ["DYLD_LIBRARY_PATH"] = "/opt/homebrew/Cellar/libomp/19.1.3/lib"

def parse_arguments():
    parser = argparse.ArgumentParser(description="Complete ML pipeline for Ensemble Classifier...")
    parser.add_argument('--path', type=str, default="/Users/shahzadali/Documents/Datasets/ADNI", help="Directory path for clinical data files")
    parser.add_argument('--file_name', type=str, default='301024_AD_data.xlsx', help="Name of the clinical data Excel file")  
    parser.add_argument('--sheet_ADNI_preprocessed', type=str, default='ADNI_CMSGT_mmse', help="Sheet name in the Excel file")   # ADNI_CMSGT_mmse_BlncDX | ADNI_CMSGT_mmse_BlncRG
    
    parser.add_argument('--output_dir', type=str, default='Project_Output_GT_testmain', help="Main project directory for output data")
    
    parser.add_argument('--scaler', type=str, default='minmax', choices=['standard', 'minmax'], help="Type of scaler to use")
    parser.add_argument('--test_size', type=float, default=0.20, help="Proportion of the dataset to include in the test split")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for data shuffling and splitting")
    parser.add_argument('--use_smote', action='store_true', help="Apply SMOTE for balancing classes in the training data")
    return parser.parse_args()


def run_command(command):
    """ Run a shell command and handle errors """
    logging.info(f"Executing command: {command}")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        logging.error(f"Command failed with return code {result.returncode}")
        raise Exception(f"Pipeline step failed: {command}")
    

def main():
    # Configure logging to display INFO level messages
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    args = parse_arguments()

    # Ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        # STEP 1: Preprocessing
        logging.info("Starting preprocessing")
        run_command(
            f"python preprocessing.py --path {args.path} "
            f"--file_name {args.file_name} --sheet_ADNI_preprocessed {args.sheet_ADNI_preprocessed} "
            f"--output_dir {args.output_dir} --scaler {args.scaler} --test_size {args.test_size} "
            f"--seed {args.seed} {'--use_smote' if args.use_smote else ''}"
        )
        
        # STEP 2: Model Training
        logging.info("Starting model training and evaluation")
        run_command(f"python ensembleCls_training.py --output_dir {args.output_dir}")

        # STEP 3: Explainability
        logging.info("Starting model explainability analysis")
        run_command(f"python explainability.py --output_dir {args.output_dir}")

    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        return

    logging.info("ML pipeline completed successfully.")

if __name__ == "__main__":
    main()
