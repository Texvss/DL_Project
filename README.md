# DL_Project

**Framework for Spoofing Detection and Evaluation based on Light CNN (LCNN)**

This repository implements an LCNN-based countermeasure for spoofing detection on the ASVspoof 2019 dataset. It provides the full pipeline: training, inference, evaluation (EER / t-DCF) and automated grading of student solutions.

## Contents

- [Installation](#installation)  
- [Configuration](#configuration)  
- [Training](#training)  
- [Inference](#inference)  
- [Evaluation](#evaluation)


## Installation

1. Clone the repository and navigate into it:
   ```bash
   git clone https://github.com/Texvss/DL_Project.git
   cd DL_Project
2. Create and activate a virtual environment (recommended):
    ```bash
    python3 -m venv venv
    source venv/bin/activate
3. Install dependencies:
    ```bash
    pip install -r requirements.txt

## Configuration
Training configurations are in src/configs/ (e.g. default.yaml).

Inference configurations are in conf/ (used by inference.py).

All parameters (data paths, model architecture, hyperparameters, CometML settings) are managed via Hydra and can be overridden on the command line:
    ```bash
    python train.py model.num_classes=2 lr=1e-3 epochs=50 dataset.train.dir=/path/to/LA_train

## Training
Run the training script:
    ```bash
    python train.py
This will instantiate the LCNN model and data loaders, configure the optimizer and scheduler, log metrics (loss, EER, accuracy), and save checkpoints.

## Inference
After training, run:
    ```bash
    python inference.py

The script will load the specified checkpoint, perform inference over the dataset, and write a CSV.

## Evaluation
To evaluate and grade student solutions, run:
    ```bash
    python grading.py

Place student CSV files (named <student>.csv) in the students_solutions/ directory first. The script will compute each student’s EER against the ASVspoof protocol and generate a grades.csv report.
