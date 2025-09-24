# Makefile for MLOps Project with MLflow and Optuna

# Variables
IMAGE_NAME = proyecto-mlops
IMAGE_TAG = latest
COMPOSE_FILE = docker-compose.yml

# Colors for terminal output
GREEN = \033[0;32m
YELLOW = \033[0;33m
RED = \033[0;31m
NC = \033[0m # No Color

.PHONY: help build train mlflow-ui predict dev clean clean-all stop logs shell

help: ## Show this help message
	@echo "$(GREEN)MLOps Project - Docker Commands$(NC)"
	@echo "======================================="
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(YELLOW)%-15s$(NC) %s\n", $$1, $$2}'

build: ## Build the Docker image
	@echo "$(GREEN)Building Docker image...$(NC)"
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) .

train: build ## Run training with MLflow and Optuna
	@echo "$(GREEN)Starting training with MLflow and Optuna...$(NC)"
	docker-compose up train

train-background: build ## Run training in background
	@echo "$(GREEN)Starting training in background...$(NC)"
	docker-compose up -d train
	@echo "$(YELLOW)Training started in background. Use 'make logs' to view progress$(NC)"

mlflow-server: ## Start MLflow server with SQLite backend (port 5000)
	@echo "$(GREEN)Starting MLflow server...$(NC)"
	docker-compose up -d mlflow-server
	@echo "$(YELLOW)MLflow server available at: http://localhost:5000$(NC)"

train-with-mlflow: build ## Run training with MLflow server
	@echo "$(GREEN)Starting MLflow server and training...$(NC)"
	docker-compose up mlflow-server train

predict: build ## Start prediction service (port 8080)
	@echo "$(GREEN)Starting prediction service...$(NC)"
	docker-compose --profile predict up predict

dev: build ## Start development container with bash
	@echo "$(GREEN)Starting development container...$(NC)"
	docker-compose --profile dev run --rm dev

shell: ## Open shell in running train container
	@echo "$(GREEN)Opening shell in train container...$(NC)"
	docker-compose exec train /bin/bash

logs: ## Show logs from all services
	docker-compose logs -f

logs-train: ## Show logs from training service only
	docker-compose logs -f train

stop: ## Stop all running containers
	@echo "$(YELLOW)Stopping all containers...$(NC)"
	docker-compose down

clean: stop ## Stop containers and remove them
	@echo "$(RED)Cleaning up containers...$(NC)"
	docker-compose down -v

clean-all: clean ## Clean everything including images
	@echo "$(RED)Removing Docker images...$(NC)"
	docker rmi $(IMAGE_NAME):$(IMAGE_TAG) 2>/dev/null || true
	@echo "$(RED)Cleaning MLflow artifacts...$(NC)"
	rm -rf src/app/train/mlruns/* 2>/dev/null || true
	rm -rf src/app/train/mlartifacts/* 2>/dev/null || true
	rm -rf models/* 2>/dev/null || true
	@echo "$(GREEN)Cleanup complete!$(NC)"

status: ## Show status of all containers
	@echo "$(GREEN)Container Status:$(NC)"
	docker-compose ps

test-local: ## Run training locally (requires Python environment)
	@echo "$(GREEN)Running training locally...$(NC)"
	python -m src.app.train.task_train

# Quick commands for common workflows
quick-train: build train-with-ui ## Build and run training with UI

quick-experiment: build ## Build and run interactive experiment
	@echo "$(GREEN)Starting experimental session...$(NC)"
	docker-compose --profile dev run --rm -v $(PWD):/app dev python -m src.app.train.task_train

monitor: ## Monitor training progress (requires running training)
	@echo "$(GREEN)Monitoring training progress...$(NC)"
	@echo "$(YELLOW)Opening MLflow UI at http://localhost:5000$(NC)"
	@sleep 2
	@command -v open >/dev/null 2>&1 && open http://localhost:5000 || echo "Please open http://localhost:5000 in your browser"

# Docker system commands
docker-prune: ## Clean up Docker system (removes unused data)
	@echo "$(YELLOW)Pruning Docker system...$(NC)"
	docker system prune -f

docker-stats: ## Show Docker container resource usage
	docker stats --no-stream

# MLflow specific commands
mlflow-list-runs: ## List all MLflow runs
	@echo "$(GREEN)Listing MLflow runs...$(NC)"
	docker-compose run --rm train mlflow runs list --experiment-id 0

mlflow-best-model: ## Show best model info
	@echo "$(GREEN)Best model information will be shown in the training output$(NC)"

# Installation check
check-deps: ## Check if required dependencies are installed
	@echo "$(GREEN)Checking dependencies...$(NC)"
	@command -v docker >/dev/null 2>&1 || { echo "$(RED)Docker is not installed$(NC)"; exit 1; }
	@command -v docker-compose >/dev/null 2>&1 || { echo "$(RED)Docker Compose is not installed$(NC)"; exit 1; }
	@echo "$(GREEN)✓ All dependencies are installed$(NC)"

# Default target
.DEFAULT_GOAL := help
