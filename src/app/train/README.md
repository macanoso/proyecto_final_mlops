# Training Module

This module contains the complete training pipeline for promotional targeting models, featuring MLOps best practices with MLflow tracking, Optuna hyperparameter optimization, and Prefect orchestration.

## 📁 Module Structure

```text
train/
├── README.md                       # This file
├── etl.py                          # Data generation and ETL processes
├── feature_engineer.py             # Feature engineering utilities
├── train.py                        # Basic training script
├── train_with_mlflow.py           # MLflow-integrated training
├── train_with_mlflow_optuna.py    # MLflow + Optuna optimization
├── task_train.py                   # Task-based training implementation
├── task_train_prefect.py          # Prefect workflow orchestration
├── run_prefect_training.py        # Prefect execution scripts
├── task_train.ipynb               # Interactive training notebook
├── backend.db                     # SQLite database for MLflow tracking
├── mlruns/                        # MLflow experiment tracking data
└── mlartifacts/                   # MLflow model artifacts storage
```

## 🚀 Features

### 1. **Data Generation & ETL** (`etl.py`)

- Synthetic user data generation for promotional targeting
- Realistic transactional and behavioral data simulation
- Customizable dataset size and characteristics
- Features include:
  - User demographics (age, registration date)
  - Transaction history (count, frequency, amounts)
  - Platform usage metrics
  - Product preferences
  - Campaign response indicators

### 2. **Feature Engineering** (`feature_engineer.py`)

- Automated feature transformation pipeline
- Handling of numerical and categorical features
- Missing value imputation strategies
- Feature scaling and encoding

### 3. **Model Training Capabilities**

#### Basic Training (`train.py`)

- Simple model training pipeline
- Quick prototyping and testing
- Minimal dependencies

#### MLflow Integration (`train_with_mlflow.py`)

- Experiment tracking and versioning
- Metric logging (accuracy, precision, recall, F1)
- Model artifact storage
- Pipeline serialization
- Support for multiple model types:
  - Logistic Regression
  - Random Forest
  - Custom sklearn estimators

#### Hyperparameter Optimization (`train_with_mlflow_optuna.py`)

- Automated hyperparameter tuning with Optuna
- Bayesian optimization strategies
- Multiple optimization metrics support:
  - Accuracy
  - F1 Score
  - Precision
  - Recall
- Customizable parameter distributions
- Integration with MLflow for tracking trials

### 4. **Workflow Orchestration** (`task_train_prefect.py`)

- Prefect-based workflow management
- Modular task design with:
  - Data generation tasks
  - Feature engineering tasks
  - Model training tasks
  - Evaluation tasks
  - Comparison tasks
- Automatic artifact creation:
  - Markdown reports
  - Performance tables
  - Confusion matrices
  - Feature importance plots
- Built-in retry logic and error handling
- Parallel model training capabilities

## 🔧 Usage

### Quick Start

#### 1. Basic Training

```python
from train import Train
from etl import UserGenerator

# Generate data
generator = UserGenerator(n_samples=10000)
df = generator.create_dataset()

# Train model
trainer = Train(
    df=df,
    numeric_features=['edad', 'total_transacciones', 'monto_total'],
    categorical_features=['tipo_usuario', 'plataforma_favorita'],
    target_column='dar_promocion'
)
pipeline = trainer.train()
```

#### 2. Training with MLflow

```python
from train_with_mlflow import TrainMlflow
import mlflow

mlflow.set_tracking_uri("sqlite:///backend.db")

trainer = TrainMlflow(
    df=df,
    numeric_features=numeric_cols,
    categorical_features=categorical_cols,
    target_column='dar_promocion',
    model=LogisticRegression(),
    mlflow_setup={
        'experiment_name': 'promotion_targeting',
        'run_name': 'logistic_regression_v1'
    }
)
pipeline, metrics = trainer.train()
```

#### 3. Hyperparameter Optimization

```python
from train_with_mlflow_optuna import TrainMlflowOptuna

trainer = TrainMlflowOptuna(
    df=df,
    numeric_features=numeric_cols,
    categorical_features=categorical_cols,
    target_column='dar_promocion',
    model_class=RandomForestClassifier,
    n_trials=50,
    optimization_metric='f1',
    param_distributions={
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 20, None],
        'min_samples_split': [2, 5, 10]
    }
)
best_pipeline, best_params = trainer.train()
```

#### 4. Prefect Workflow Execution

```bash
# Quick demo
python run_prefect_training.py quick

# Single model training with optimization
python run_prefect_training.py single

# Model comparison
python run_prefect_training.py compare
```

### Advanced Workflows

#### Custom Prefect Flow

```python
from task_train_prefect import train_model_flow, compare_models_flow

# Train single model with specific parameters
pipeline, run_id = train_model_flow(
    n_samples=5000,
    model_type="RandomForest",
    n_trials=30,
    optimization_metric="f1"
)

# Compare multiple models
results = compare_models_flow(
    n_samples=5000,
    n_trials=20
)
```

## 📊 Monitoring & Visualization

### MLflow UI

Start the MLflow tracking server:

```bash
mlflow ui --backend-store-uri sqlite:///backend.db
```

Access at: <http://localhost:5000>

Features available:

- Experiment comparison
- Metric visualization
- Model registry
- Artifact browsing

### Prefect UI

Start the Prefect server:

```bash
prefect server start
```

Access at: <http://localhost:4200>

Features available:

- Flow run monitoring
- Task execution graphs
- Artifact visualization
- Error tracking and logs

## 📈 Metrics & Evaluation

The training pipeline tracks and logs the following metrics:

- **Accuracy**: Overall prediction accuracy
- **Precision**: True positive rate
- **Recall**: Sensitivity/hit rate
- **F1 Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed classification breakdown
- **Feature Importance**: For tree-based models
- **ROC-AUC**: Area under the ROC curve (when applicable)

## 🔄 Model Versioning

Models are automatically versioned through MLflow with:

- Unique run IDs
- Timestamp tracking
- Git commit hash (when in git repo)
- Training parameters
- Dataset fingerprints
- Performance metrics

## 📝 Configuration

### Environment Variables

```bash
# MLflow tracking
export MLFLOW_TRACKING_URI="sqlite:///backend.db"
export MLFLOW_EXPERIMENT_NAME="promotion_targeting"

# Prefect configuration
export PREFECT_API_URL="http://localhost:4200/api"
```

### Model Parameters

Default hyperparameter ranges for optimization are defined in `train_with_mlflow_optuna.py` but can be customized:

```python
param_distributions = {
    'LogisticRegression': {
        'C': (0.001, 10.0),  # log scale
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga']
    },
    'RandomForest': {
        'n_estimators': (50, 300),
        'max_depth': (5, 50),
        'min_samples_split': (2, 20)
    }
}
```

## 🧪 Testing

Run the interactive notebook for experimentation:

```bash
jupyter notebook task_train.ipynb
```

## 📦 Dependencies

- **scikit-learn**: Machine learning models and pipelines
- **pandas**: Data manipulation
- **numpy**: Numerical operations
- **mlflow**: Experiment tracking and model management
- **optuna**: Hyperparameter optimization
- **prefect**: Workflow orchestration
- **matplotlib/seaborn**: Visualization (for artifacts)

## 🔍 Directory Details

### `mlruns/`

Contains MLflow tracking data:

- Experiment metadata
- Run information
- Logged metrics
- Parameter values

### `mlartifacts/`

Stores model artifacts:

- Serialized models (pickle/joblib)
- Model signatures
- Requirements files
- Custom artifacts (plots, reports)

## 🚦 Best Practices

1. **Always use MLflow tracking** for production training
2. **Run hyperparameter optimization** for new model types
3. **Use Prefect workflows** for complex training pipelines
4. **Version your models** with meaningful tags
5. **Monitor metrics** across experiments
6. **Document parameter choices** in MLflow run notes
7. **Clean up old artifacts** periodically to save space

## 📚 Additional Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Optuna Documentation](https://optuna.readthedocs.io/)
- [Prefect Documentation](https://docs.prefect.io/)
- [Scikit-learn Pipeline Guide](https://scikit-learn.org/stable/modules/compose.html)

## 🤝 Contributing

When adding new training capabilities:

1. Implement MLflow tracking
2. Add Optuna optimization support
3. Create Prefect tasks for orchestration
4. Update this README
5. Add example usage in notebooks

## 📧 Support

For questions or issues related to the training module, please refer to the main project documentation or contact the MLOps team.
