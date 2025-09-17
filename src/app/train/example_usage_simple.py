"""
Example of how to use the simplified TrainMlflowOptuna class
where you directly pass the parameter distributions.
"""

import pandas as pd
import numpy as np
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from train_with_mlflow_optuna import TrainMlflowOptuna


def example_logistic_regression():
    """Example using LogisticRegression with custom parameter distributions."""
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    df = pd.DataFrame({
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples),
        'feature3': np.random.randn(n_samples),
        'category': np.random.choice(['A', 'B', 'C'], n_samples),
        'target': np.random.choice([0, 1], n_samples)
    })
    
    # Define parameter distributions for LogisticRegression
    param_distributions = {
        'C': ('float', 0.001, 100, True),  # (type, min, max, log_scale)
        'penalty': ('categorical', ['l1', 'l2']),
        'max_iter': ('int', 100, 1000),
        'solver': ('categorical', ['liblinear', 'saga'])  # Compatible with both l1 and l2
    }
    
    # Initialize trainer
    trainer = TrainMlflowOptuna(
        df=df,
        numeric_features=['feature1', 'feature2', 'feature3'],
        categorical_features=['category'],
        target_column='target',
        model_class=LogisticRegression,
        test_size=0.2,
        n_trials=10,
        optimization_metric='accuracy',
        param_distributions=param_distributions,
        model_params={'random_state': 42}  # Fixed parameters
    )
    
    # Run training with Optuna optimization
    best_pipeline, run_id, study = trainer.train()
    
    print("\nExample completed!")
    return best_pipeline, run_id, study


def example_random_forest():
    """Example using RandomForest with custom parameter distributions."""
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    df = pd.DataFrame({
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples),
        'feature3': np.random.randn(n_samples),
        'feature4': np.random.randn(n_samples),
        'category1': np.random.choice(['X', 'Y', 'Z'], n_samples),
        'category2': np.random.choice(['P', 'Q'], n_samples),
        'target': np.random.choice([0, 1, 2], n_samples)  # Multiclass
    })
    
    # Define parameter distributions for RandomForest
    param_distributions = {
        'n_estimators': ('int', 50, 200),
        'max_depth': ('int', 5, 30),
        'min_samples_split': ('int', 2, 10),
        'min_samples_leaf': ('int', 1, 5),
        'max_features': ('categorical', ['sqrt', 'log2', None])
    }
    
    # Initialize trainer
    trainer = TrainMlflowOptuna(
        df=df,
        numeric_features=['feature1', 'feature2', 'feature3', 'feature4'],
        categorical_features=['category1', 'category2'],
        target_column='target',
        model_class=RandomForestClassifier,
        test_size=0.3,
        n_trials=15,
        optimization_metric='f1',  # Optimize for F1 score
        param_distributions=param_distributions,
        model_params={'random_state': 42, 'n_jobs': -1}  # Fixed parameters
    )
    
    # Run training
    best_pipeline, run_id, study = trainer.train()
    
    print("\nExample completed!")
    return best_pipeline, run_id, study


def example_with_roc_auc():
    """Example optimizing for ROC-AUC (requires probability predictions)."""
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    df = pd.DataFrame({
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples),
        'target': np.random.choice([0, 1], n_samples)
    })
    
    # Simple parameter distributions
    param_distributions = {
        'C': ('float', 0.1, 10, True),
        'penalty': ('categorical', ['l2'])  # l2 for compatibility with default solver
    }
    
    # Initialize trainer optimizing for ROC-AUC
    trainer = TrainMlflowOptuna(
        df=df,
        numeric_features=['feature1', 'feature2'],
        categorical_features=[],
        target_column='target',
        model_class=LogisticRegression,
        test_size=0.2,
        n_trials=8,
        optimization_metric='roc_auc',  # Optimize for ROC-AUC
        param_distributions=param_distributions,
        model_params={'random_state': 42}
    )
    
    # Run training
    best_pipeline, run_id, study = trainer.train()
    
    print("\nExample completed!")
    return best_pipeline, run_id, study


if __name__ == "__main__":
    # Set up MLflow
    mlflow.set_experiment("optuna_simple_examples")
    mlflow.sklearn.autolog()
    
    print("=" * 50)
    print("Running Logistic Regression Example")
    print("=" * 50)
    example_logistic_regression()
    
    print("\n" + "=" * 50)
    print("Running Random Forest Example")
    print("=" * 50)
    example_random_forest()
    
    print("\n" + "=" * 50)
    print("Running ROC-AUC Optimization Example")
    print("=" * 50)
    example_with_roc_auc()
    
    print("\n\nAll examples completed!")
    print(f"View results in MLflow UI: mlflow ui")
