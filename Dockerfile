# Use Python 3.11 slim image as base
FROM python:3.11

# Set working directory
WORKDIR /app

# Install system dependencies that might be needed for ML libraries
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
# Using the standalone installer method
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy the project files
COPY pyproject.toml uv.lock ./

# Install dependencies using uv
# First, install all dependencies from pyproject.toml
RUN uv pip install --system -r pyproject.toml

# Ensure we have the correct MLflow version with all components using uv
# This will respect the dependency resolver
RUN uv pip install --system --upgrade --force-reinstall \
    mlflow==3.4.0 \
    mlflow-skinny==3.4.0 \
    mlflow-tracing==3.4.0 \
    gunicorn

# Copy the rest of the application code
COPY src/ ./src/
COPY notebooks/ ./notebooks/

# Create directories for MLflow artifacts and data
RUN mkdir -p /app/data /app/mlruns /app/mlartifacts /app/models /app/mlflow

# Set Python path
ENV PYTHONPATH=/app

# Expose ports
EXPOSE 5000 8888

# Default command - can be overridden when running the container
CMD ["python", "-m", "src.app.train.task_train"]
