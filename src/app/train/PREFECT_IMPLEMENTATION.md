# Prefect Implementation with Optuna & MLflow

## Overview

The Prefect implementation orchestrates the complete machine learning pipeline with hyperparameter optimization using Optuna and experiment tracking with MLflow.

## Key Components

### 1. Tasks

- **`task_generate_data`**: Generates synthetic training data
- **`task_feature_engineering`**: Applies feature transformations
- **`task_train_with_optuna`**: Trains models with Optuna hyperparameter optimization
- **`task_create_model_report`**: Creates comprehensive training reports

### 2. Flows

- **`train_model_flow`**: Main flow for training a single model
- **`compare_models_flow`**: Compares multiple models and optimization metrics

### 3. Artifacts Created

#### Table Artifacts

- **Data Generation Summary**: Dataset statistics
- **Feature Engineering Summary**: Feature transformation results
- **Optuna Trials Summary**: Top trials with scores and parameters
- **Best Hyperparameters**: Optimal parameters found
- **Model Comparison Results**: Side-by-side model comparison
- **Training Flow Summary**: Overall flow execution summary

#### Markdown Artifacts

- **Model Training Report**: Comprehensive report with:
  - Training summary
  - Optimization results
  - Best hyperparameters
  - Validation metrics
  - Top trials
  - Usage instructions

## Usage Examples

### Basic Training

```python
from task_train_prefect import train_model_flow

# Train a LogisticRegression model
pipeline, run_id = train_model_flow(
    n_samples=10000,
    model_type="LogisticRegression",
    n_trials=20,
    optimization_metric="accuracy"
)
```

### Model Comparison

```python
from task_train_prefect import compare_models_flow

# Compare multiple models
results = compare_models_flow(
    n_samples=10000,
    n_trials=15
)
```

## Features

### 1. Automatic Retry Logic

- All tasks have retry configurations
- Automatic retry on failures with delays

### 2. Comprehensive Logging

- Detailed logging at each step
- Progress tracking for long-running tasks

### 3. Artifact Tracking

- Automatic creation of table artifacts for data inspection
- Markdown reports for documentation
- All artifacts viewable in Prefect UI

### 4. MLflow Integration

- Automatic MLflow experiment tracking
- Model versioning and storage
- Metric logging for all trials

### 5. Optuna Optimization

- Flexible parameter distributions
- Multiple optimization metrics supported
- Trial history tracking

## Viewing Results

### 1. Start Prefect Server

```bash
prefect server start
```

Navigate to: <http://localhost:4200>

### 2. Start MLflow UI

```bash
mlflow ui
```

Navigate to: <http://localhost:5000>

### 3. Run Training

```bash
# Quick demo
python run_prefect_training.py quick

# Single model training
python run_prefect_training.py single

# Model comparison
python run_prefect_training.py compare
```

## Configuration

### Parameter Distributions

Define custom parameter search spaces:

```python
param_distributions = {
    'C': ('float', 0.001, 100, True),  # log scale
    'penalty': ('categorical', ['l1', 'l2']),
    'max_iter': ('int', 100, 2000)
}
```

### Supported Models

- LogisticRegression
- RandomForestClassifier
- Easy to extend for other scikit-learn models

### Optimization Metrics

- accuracy
- f1
- precision
- recall
- roc_auc

## Flow Diagram

```
┌─────────────────┐
│ Generate Data   │
└────────┬────────┘
         ↓
┌─────────────────┐
│ Feature Eng.    │
└────────┬────────┘
         ↓
┌─────────────────┐
│ Optuna Training │──→ MLflow Tracking
└────────┬────────┘
         ↓
┌─────────────────┐
│ Create Report   │──→ Prefect Artifacts
└─────────────────┘
```

## Benefits

1. **Reproducibility**: All parameters and results tracked
2. **Scalability**: Easy to add new models or metrics
3. **Observability**: Complete visibility through Prefect UI
4. **Optimization**: Automatic hyperparameter tuning
5. **Documentation**: Automatic report generation

## Next Steps

1. Deploy to Prefect Cloud for production
2. Add more model types (XGBoost, LightGBM)
3. Implement cross-validation in Optuna trials
4. Add data validation tasks
5. Create deployment flows for model serving
