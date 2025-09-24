"""
Prefect workflow for training models with Optuna hyperparameter optimization and MLflow tracking.
"""

import json
from datetime import datetime

import mlflow
import numpy as np
import pandas as pd
from etl import UserGenerator
from feature_engineer import FeatureEngineer
from prefect import flow, get_run_logger, task
from prefect.artifacts import create_markdown_artifact, create_table_artifact
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from train_with_mlflow_optuna import TrainMlflowOptuna


@task(name="Generate_Data", retries=2, retry_delay_seconds=10)
def task_generate_data(n_samples: int = 10000) -> pd.DataFrame:
    """
    Generate synthetic user data for training.
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        Generated dataframe
    """
    logger = get_run_logger()
    logger.info(f"Generating {n_samples} samples...")
    
    user_generator = UserGenerator(n_samples=n_samples)
    df = user_generator.create_dataset()
    
    # Create summary artifact
    summary_df = pd.DataFrame({
        'Metric': ['Total Samples', 'Total Features', 'Missing Values', 'Target Distribution'],
        'Value': [
            len(df),
            len(df.columns),
            df.isnull().sum().sum(),
            df['dar_promocion'].value_counts().to_dict() if 'dar_promocion' in df.columns else 'N/A'
        ]
    })
    
    create_table_artifact(
        key="data-generation-summary",
        table=summary_df.to_dict(orient='records'),
        description=f"Data Generation Summary - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    
    logger.info(f"Generated {len(df)} samples with {len(df.columns)} columns")
    return df


@task(name="Feature_Engineering", retries=2, retry_delay_seconds=10)
def task_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply feature engineering to the dataset.
    
    Args:
        df: Input dataframe
        
    Returns:
        Feature-engineered dataframe
    """
    logger = get_run_logger()
    logger.info("Starting feature engineering...")
    
    initial_columns = len(df.columns)
    feature_engineer = FeatureEngineer(df)
    df_engineered = feature_engineer.create_features()
    
    # Create feature engineering summary
    feature_summary = pd.DataFrame({
        'Metric': ['Initial Features', 'Final Features', 'Features Added', 'Dataset Size'],
        'Value': [
            initial_columns,
            len(df_engineered.columns),
            len(df_engineered.columns) - initial_columns,
            f"{len(df_engineered)} rows"
        ]
    })
    
    create_table_artifact(
        key="feature-engineering-summary",
        table=feature_summary.to_dict(orient='records'),
        description="Feature Engineering Summary"
    )
    
    logger.info(f"Feature engineering complete: {initial_columns} -> {len(df_engineered.columns)} features")
    return df_engineered


@task(name="Train_Model_Optuna", retries=1, retry_delay_seconds=30)
def task_train_with_optuna(
    df: pd.DataFrame,
    model_type: str = "LogisticRegression",
    n_trials: int = 20,
    optimization_metric: str = "accuracy"
) -> tuple:
    """
    Train model with Optuna hyperparameter optimization and MLflow tracking.
    
    Args:
        df: Feature-engineered dataframe
        model_type: Type of model to train ('LogisticRegression' or 'RandomForest')
        n_trials: Number of Optuna trials
        optimization_metric: Metric to optimize
        
    Returns:
        Tuple of (best_pipeline, best_run_id, study, metrics_dict)
    """
    logger = get_run_logger()
    logger.info(f"Starting Optuna optimization for {model_type} with {n_trials} trials...")
    
    # Define features
    numeric_features = [
        'days_since_registration', 'total_purchases', 'avg_order_value',
        'last_purchase_days', 'sessions_last_30_days', 'time_on_site_minutes',
        'pages_per_session', 'cart_abandonment_rate', 'purchase_frequency'
    ]
    categorical_features = ['age_group', 'location', 'device_type', 'subscription_type']
    target_column = 'dar_promocion'
    
    # Define parameter distributions based on model type
    if model_type == "LogisticRegression":
        model_class = LogisticRegression
        param_distributions = {
            'C': ('float', 0.001, 100, True),
            'penalty': ('categorical', ['l1', 'l2']),
            'max_iter': ('int', 200, 2000),
            'solver': ('categorical', ['liblinear', 'saga'])
        }
        fixed_params = {'random_state': 42}
    elif model_type == "RandomForest":
        model_class = RandomForestClassifier
        param_distributions = {
            'n_estimators': ('int', 50, 200),
            'max_depth': ('int', 5, 30),
            'min_samples_split': ('int', 2, 15),
            'min_samples_leaf': ('int', 1, 10),
            'max_features': ('categorical', ['sqrt', 'log2'])
        }
        fixed_params = {'random_state': 42, 'n_jobs': -1}
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Set up MLflow
    mlflow.set_experiment(f"prefect_{model_type.lower()}_training")
    mlflow.sklearn.autolog()
    
    # Create trainer
    trainer = TrainMlflowOptuna(
        df=df,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        target_column=target_column,
        model_class=model_class,
        test_size=0.25,
        n_trials=n_trials,
        optimization_metric=optimization_metric,
        param_distributions=param_distributions,
        model_params=fixed_params
    )
    
    # Run optimization
    best_pipeline, best_run_id, study = trainer.train()
    
    # Create Optuna trials table artifact
    trials_data = []
    for trial in study.trials:
        trials_data.append({
            'Trial': trial.number,
            'Value': f"{trial.value:.4f}" if trial.value else "Failed",
            'State': trial.state.name,
            'Duration (s)': f"{(trial.datetime_complete - trial.datetime_start).total_seconds():.2f}" 
                           if trial.datetime_complete else "N/A",
            'Parameters': json.dumps(trial.params, indent=2)[:100] + "..."  # Truncate for display
        })
    
    trials_df = pd.DataFrame(trials_data)
    
    create_table_artifact(
        key="optuna-trials-summary",
        table=trials_df.head(10).to_dict(orient='records'),  # Show top 10 trials
        description=f"Optuna Optimization Results - {model_type} - Best {optimization_metric}: {study.best_value:.4f}"
    )
    
    # Create best parameters artifact
    best_params_df = pd.DataFrame([
        {'Parameter': k, 'Value': v} for k, v in study.best_params.items()
    ])
    
    create_table_artifact(
        key="best-hyperparameters",
        table=best_params_df.to_dict(orient='records'),
        description=f"Best Hyperparameters for {model_type}"
    )
    
    # Calculate validation metrics
    X_test = df.drop(columns=[target_column])[:100]  # Sample for quick validation
    y_test = df[target_column][:100]
    y_pred = best_pipeline.predict(X_test)
    
    metrics_dict = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1': f1_score(y_test, y_pred, average='weighted')
    }
    
    logger.info(f"Optimization complete! Best {optimization_metric}: {study.best_value:.4f}")
    logger.info(f"Best parameters: {study.best_params}")
    logger.info(f"MLflow Run ID: {best_run_id}")
    
    return best_pipeline, best_run_id, study, metrics_dict


@task(name="Create_Model_Report", retries=1)
def task_create_model_report(
    model_type: str,
    best_run_id: str,
    study,
    metrics_dict: dict,
    n_trials: int
) -> None:
    """
    Create comprehensive model training report as markdown artifact.
    
    Args:
        model_type: Type of model trained
        best_run_id: MLflow run ID
        study: Optuna study object
        metrics_dict: Dictionary of validation metrics
        n_trials: Number of trials performed
    """
    logger = get_run_logger()
    logger.info("Creating model training report...")
    
    # Create markdown report
    markdown_content = f"""
# Model Training Report - {model_type}

## Training Summary
- **Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Model Type**: {model_type}
- **Number of Trials**: {n_trials}
- **MLflow Run ID**: `{best_run_id}`

## Optimization Results
- **Best Score**: {study.best_value:.4f}
- **Optimization Metric**: {study.trials[0].user_attrs.get('metric_name', 'accuracy') if study.trials else 'N/A'}
- **Total Trials Completed**: {len([t for t in study.trials if t.state.name == 'COMPLETE'])}

## Best Hyperparameters
```python
{json.dumps(study.best_params, indent=2)}
```

## Validation Metrics
| Metric | Score |
|--------|-------|
| Accuracy | {metrics_dict['accuracy']:.4f} |
| Precision | {metrics_dict['precision']:.4f} |
| Recall | {metrics_dict['recall']:.4f} |
| F1 Score | {metrics_dict['f1']:.4f} |

## Top 5 Trials
| Trial | Score | Parameters |
|-------|-------|------------|
"""
    
    # Add top 5 trials
    sorted_trials = sorted(study.trials, key=lambda t: t.value if t.value else 0, reverse=True)[:5]
    for trial in sorted_trials:
        if trial.value:
            params_str = ', '.join([f"{k}={v}" for k, v in list(trial.params.items())[:3]])
            markdown_content += f"| {trial.number} | {trial.value:.4f} | {params_str}... |\n"
    
    markdown_content += f"""

## How to Use the Model
```python
import mlflow

# Load the best model
model = mlflow.sklearn.load_model(f"runs:/{best_run_id}/model")

# Make predictions
predictions = model.predict(X_new)
```

## Next Steps
1. Review the model performance in MLflow UI
2. Deploy the model if metrics meet requirements
3. Monitor model performance in production
"""
    
    create_markdown_artifact(
        key="model-training-report",
        markdown=markdown_content,
        description=f"Complete Training Report for {model_type}"
    )
    
    logger.info("Model training report created successfully")


@flow(name="Train_Model_With_Optuna", log_prints=True)
def train_model_flow(
    n_samples: int = 10000,
    model_type: str = "LogisticRegression",
    n_trials: int = 20,
    optimization_metric: str = "accuracy"
):
    """
    Main Prefect flow for training models with Optuna optimization.
    
    Args:
        n_samples: Number of samples to generate
        model_type: Type of model to train
        n_trials: Number of Optuna trials
        optimization_metric: Metric to optimize
        
    Returns:
        Trained pipeline and MLflow run ID
    """
    logger = get_run_logger()
    logger.info(f"Starting training flow for {model_type}")
    
    # Task 1: Generate data
    df = task_generate_data(n_samples)
    
    # Task 2: Feature engineering
    df_engineered = task_feature_engineering(df)
    
    # Task 3: Train with Optuna
    best_pipeline, best_run_id, study, metrics_dict = task_train_with_optuna(
        df_engineered,
        model_type,
        n_trials,
        optimization_metric
    )
    
    # Task 4: Create report
    task_create_model_report(
        model_type,
        best_run_id,
        study,
        metrics_dict,
        n_trials
    )
    
    logger.info(f"Training flow complete! Best model saved with run ID: {best_run_id}")
    
    # Create final summary artifact
    final_summary = pd.DataFrame({
        'Metric': ['Model Type', 'Best Score', 'MLflow Run ID', 'Total Time'],
        'Value': [
            model_type,
            f"{study.best_value:.4f}",
            best_run_id,
            f"{sum((t.datetime_complete - t.datetime_start).total_seconds() for t in study.trials if t.datetime_complete):.2f}s"
        ]
    })
    
    create_table_artifact(
        key="training-flow-summary",
        table=final_summary.to_dict(orient='records'),
        description="Final Training Flow Summary"
    )
    
    return best_pipeline, best_run_id


@flow(name="Compare_Models", log_prints=True)
def compare_models_flow(
    n_samples: int = 10000,
    n_trials: int = 15
):
    """
    Flow to compare multiple models with Optuna optimization.
    
    Args:
        n_samples: Number of samples to generate
        n_trials: Number of Optuna trials per model
        
    Returns:
        Dictionary with results for each model
    """
    logger = get_run_logger()
    logger.info("Starting model comparison flow...")
    
    # Generate data once
    df = task_generate_data(n_samples)
    df_engineered = task_feature_engineering(df)
    
    results = {}
    models_to_compare = ["LogisticRegression", "RandomForest"]
    metrics_to_try = ["accuracy", "f1"]
    
    comparison_data = []
    
    for model_type in models_to_compare:
        for metric in metrics_to_try:
            logger.info(f"Training {model_type} optimizing for {metric}...")
            
            best_pipeline, best_run_id, study, metrics_dict = task_train_with_optuna(
                df_engineered,
                model_type,
                n_trials,
                metric
            )
            
            comparison_data.append({
                'Model': model_type,
                'Optimization Metric': metric,
                'Best Score': f"{study.best_value:.4f}",
                'Accuracy': f"{metrics_dict['accuracy']:.4f}",
                'F1 Score': f"{metrics_dict['f1']:.4f}",
                'MLflow Run ID': best_run_id[:8] + "..."
            })
            
            results[f"{model_type}_{metric}"] = {
                'pipeline': best_pipeline,
                'run_id': best_run_id,
                'best_score': study.best_value
            }
    
    # Create comparison table
    comparison_df = pd.DataFrame(comparison_data)
    
    create_table_artifact(
        key="model-comparison-results",
        table=comparison_df.to_dict(orient='records'),
        description="Model Comparison Results - Multiple Models and Metrics"
    )
    
    # Find best overall model
    best_model_key = max(results.keys(), key=lambda k: results[k]['best_score'])
    
    logger.info(f"Model comparison complete! Best model: {best_model_key}")
    
    return results


if __name__ == "__main__":
    # Example 1: Train a single model
    pipeline, run_id = train_model_flow(
        n_samples=5000,
        model_type="LogisticRegression",
        n_trials=10,
        optimization_metric="accuracy"
    )
    
    # Example 2: Compare multiple models
    # results = compare_models_flow(n_samples=5000, n_trials=10)