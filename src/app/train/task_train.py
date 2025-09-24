import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.app.train.etl import UserGenerator
from src.app.train.feature_engineer import FeatureEngineer
from src.app.train.train_with_mlflow_optuna import TrainMlflowOptuna


def task_train():
    # Set up MLflow - use environment variable or default to local server
    import os
    import time

    import requests
    
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
    
    # Wait for MLflow server to be ready
    max_retries = 30
    for i in range(max_retries):
        try:
            # Try to connect to the MLflow tracking server API
            response = requests.get(f"{mlflow_uri}/api/2.0/mlflow/experiments/list", timeout=5)
            if response.status_code in [200, 401, 403]:  # Any response means server is up
                print(f"MLflow server is ready at: {mlflow_uri}")
                break
        except Exception as e:
            if i < max_retries - 1:
                print(f"Waiting for MLflow server... (attempt {i+1}/{max_retries})")
                time.sleep(2)
            else:
                print(f"Error details: {str(e)}")
                raise Exception(f"Could not connect to MLflow server at {mlflow_uri} after {max_retries} attempts")
    
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("promociones_targeting_optuna")
    
    print(f"Connected to MLflow server at: {mlflow_uri}")
    
    # Generate data
    user_generator = UserGenerator(n_samples=25000)
    df = user_generator.create_dataset()

    # Feature engineering
    feature_engineer = FeatureEngineer(df)
    df_engineered = feature_engineer.create_features()

    # Define features and target
    numeric_features = ['days_since_registration', 'total_purchases', 'avg_order_value', 
                   'last_purchase_days', 'sessions_last_30_days', 'time_on_site_minutes', 
                   'pages_per_session', 'cart_abandonment_rate', 'purchase_frequency',
                   'total_purchases_per_day', 'days_between_first_and_last_purchase']
    categorical_features = ['age_group', 'location', 'device_type', 'subscription_type', 
                           'bucket_avg_order_value']
    target_column = 'dar_promocion'
    
    # Define hyperparameter search space for Logistic Regression
    params_random_forest = {
    'n_estimators': ('int', 50, 200),
    'max_depth': ('int', 5, 30),
    'min_samples_split': ('int', 2, 10),
    'min_samples_leaf': ('int', 1, 5),
    'max_features': ('categorical', ['sqrt', 'log2', None])
}
    
    # Train with MLflow + Optuna
    print("=" * 80)
    print("Starting MLflow + Optuna Training Pipeline")
    print("=" * 80)
    
    trainer = TrainMlflowOptuna(
        df=df_engineered,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        target_column=target_column,
        model_class=RandomForestClassifier,
        test_size=0.25,
        model_params={'random_state': 42, 'n_jobs': -1},  # Fixed parameters
        n_trials=1,  # Number of Optuna trials
        optimization_metric='accuracy',
        param_distributions=params_random_forest
    )
    
    # Execute training with hyperparameter optimization
    best_pipeline, best_run_id, study = trainer.train()
    
    print("=" * 80)
    print(f"Training completed! Best model saved with run_id: {best_run_id}")
    print(f"Access MLflow UI at: http://localhost:5001")
    print("=" * 80)
    
    return best_pipeline, best_run_id


if __name__ == "__main__":
    task_train()