"""Example client for User Promotion Prediction API

This script demonstrates how to interact with the Flask API endpoints.

Usage:
    python client_example.py
"""

import json
import requests
import time
from typing import Dict, List, Any


BASE_URL = "http://localhost:9696"


def check_health() -> Dict:
    """Check if the API is healthy and running."""
    response = requests.get(f"{BASE_URL}/health")
    return response.json()


def get_model_info() -> Dict:
    """Get information about the loaded model."""
    response = requests.get(f"{BASE_URL}/model_info")
    return response.json()


def predict_single_user(user_data: Dict) -> Dict:
    """
    Make a prediction for a single user.
    
    Args:
        user_data: Dictionary with user features
    
    Returns:
        Prediction result
    """
    response = requests.post(
        f"{BASE_URL}/predict",
        json=user_data,
        headers={"Content-Type": "application/json"}
    )
    return response.json()


def predict_batch_users(users_data: List[Dict]) -> Dict:
    """
    Make predictions for multiple users.
    
    Args:
        users_data: List of dictionaries with user features
    
    Returns:
        Batch prediction results
    """
    response = requests.post(
        f"{BASE_URL}/predict_batch",
        json={"users": users_data},
        headers={"Content-Type": "application/json"}
    )
    return response.json()


def generate_and_predict(n_samples: int = 100, seed: int = 42) -> Dict:
    """
    Generate synthetic users and predict.
    
    Args:
        n_samples: Number of synthetic users to generate
        seed: Random seed for reproducibility
    
    Returns:
        Predictions for synthetic users
    """
    response = requests.post(
        f"{BASE_URL}/predict_synthetic",
        json={"n_samples": n_samples, "seed": seed},
        headers={"Content-Type": "application/json"}
    )
    return response.json()


def export_predictions(n_samples: int = 1000, include_probabilities: bool = False) -> Dict:
    """
    Export predictions to CSV file.
    
    Args:
        n_samples: Number of users to generate and predict
        include_probabilities: Whether to include probability columns
    
    Returns:
        Export status
    """
    response = requests.post(
        f"{BASE_URL}/export_predictions",
        json={
            "n_samples": n_samples,
            "include_probabilities": include_probabilities
        },
        headers={"Content-Type": "application/json"}
    )
    return response.json()


def main():
    """Run example API calls."""
    
    print("=" * 60)
    print("USER PROMOTION PREDICTION API - CLIENT EXAMPLE")
    print("=" * 60)
    
    # 1. Check health
    print("\n1. Checking API health...")
    try:
        health = check_health()
        print(f"   Status: {health['status']}")
        print(f"   Model loaded: {health['model_loaded']}")
        print(f"   Available endpoints: {len(health['endpoints'])}")
    except requests.ConnectionError:
        print("   ❌ Could not connect to API. Make sure it's running on port 9696")
        return
    
    # 2. Get model info
    print("\n2. Getting model information...")
    model_info = get_model_info()
    print(f"   Model type: {model_info.get('model_type', 'Unknown')}")
    print(f"   Number of features: {model_info.get('n_features', 'Unknown')}")
    if 'n_estimators' in model_info:
        print(f"   Number of estimators: {model_info['n_estimators']}")
    
    # 3. Single user prediction
    print("\n3. Making single user prediction...")
    sample_user = {
        "user_id": "USER-TEST-001",
        "age_group": "26-35",
        "location": "Buenos Aires",
        "device_type": "Mobile",
        "subscription_type": "Premium",
        "days_since_registration": 180,
        "total_purchases": 15,
        "avg_order_value": 150.75,
        "last_purchase_days": 5,
        "sessions_last_30_days": 20,
        "time_on_site_minutes": 45.5,
        "pages_per_session": 8.2,
        "cart_abandonment_rate": 0.15,
        "purchase_frequency": 2.5
    }
    
    single_result = predict_single_user(sample_user)
    print(f"   User ID: {single_result['user_id']}")
    print(f"   Should receive promotion: {single_result['should_receive_promotion']}")
    print(f"   Confidence: {single_result['confidence']:.2%}")
    print(f"   Probability of promotion: {single_result['probabilities']['promotion']:.2%}")
    
    # 4. Batch prediction
    print("\n4. Making batch prediction for 3 users...")
    batch_users = [
        {
            "user_id": f"USER-BATCH-00{i}",
            "age_group": ["18-25", "26-35", "36-45"][i-1],
            "location": ["Cordoba", "Rosario", "Mendoza"][i-1],
            "device_type": ["Mobile", "Desktop", "Tablet"][i-1],
            "subscription_type": ["Free", "Basic", "Premium"][i-1],
            "days_since_registration": 60 * i,
            "total_purchases": 5 * i,
            "avg_order_value": 50.0 * i,
            "last_purchase_days": 30 - (5 * i),
            "sessions_last_30_days": 10 * i,
            "time_on_site_minutes": 15.0 * i,
            "pages_per_session": 3.0 * i,
            "cart_abandonment_rate": 0.1 * i,
            "purchase_frequency": 0.5 * i
        }
        for i in range(1, 4)
    ]
    
    batch_result = predict_batch_users(batch_users)
    print(f"   Total users: {batch_result['total_users']}")
    print(f"   Users to target: {batch_result['users_to_target']}")
    print(f"   Target percentage: {batch_result['target_percentage']}%")
    
    for pred in batch_result['predictions']:
        print(f"   - {pred['user_id']}: {pred['should_receive_promotion']} "
              f"(confidence: {pred['confidence']:.2%})")
    
    # 5. Synthetic data prediction
    print("\n5. Generating and predicting synthetic users...")
    synthetic_result = generate_and_predict(n_samples=50, seed=123)
    print(f"   Generated users: {synthetic_result['generation_info']['n_samples_generated']}")
    print(f"   Users to target: {synthetic_result['users_to_target']}")
    print(f"   Target percentage: {synthetic_result['target_percentage']}%")
    
    # Show first 3 predictions
    print("   Sample predictions:")
    for pred in synthetic_result['predictions'][:3]:
        print(f"   - {pred['user_id']}: {pred['should_receive_promotion']} "
              f"(prob: {pred['probabilities']['promotion']:.2%})")
    
    # 6. Export predictions (smaller sample for demo)
    print("\n6. Exporting predictions to CSV...")
    export_result = export_predictions(n_samples=100, include_probabilities=True)
    print(f"   File saved: {export_result['file_saved']}")
    print(f"   Total records: {export_result['total_records']}")
    print(f"   Users to target: {export_result['users_to_target']} "
          f"({export_result['target_percentage']}%)")
    
    print("\n" + "=" * 60)
    print("CLIENT EXAMPLE COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()
