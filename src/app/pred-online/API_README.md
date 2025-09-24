# User Promotion Targeting Prediction API

Flask-based REST API for predicting which users should receive promotional offers using a Random Forest model.

## 🚀 Quick Start

### Installation

1. Install dependencies:
```bash
pip install -r requirements_api.txt
```

2. Ensure the model file exists at `models/model_rf.pkl`

3. Start the Flask server:
```bash
python api.py
```

The API will be available at `http://localhost:9696`

### Test the API

Check if the API is running:
```bash
curl http://localhost:9696/health
```

## 📋 API Endpoints

### 1. Health Check
**GET** `/health`

Check service status and available endpoints.

```bash
curl http://localhost:9696/health
```

**Response:**
```json
{
    "status": "healthy",
    "model_loaded": true,
    "model_type": "RandomForestClassifier",
    "service": "User Promotion Targeting Prediction",
    "version": "1.0",
    "endpoints": {...}
}
```

### 2. Model Information
**GET** `/model_info`

Get details about the loaded model.

```bash
curl http://localhost:9696/model_info
```

**Response:**
```json
{
    "model_type": "RandomForestClassifier",
    "n_features": 16,
    "n_estimators": 100,
    "max_depth": 10,
    "top_features": [...]
}
```

### 3. Single User Prediction
**POST** `/predict`

Predict promotion targeting for a single user.

**Request:**
```bash
curl -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "USER-001",
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
  }'
```

**Response:**
```json
{
    "user_id": "USER-001",
    "should_receive_promotion": "Si",
    "confidence": 0.85,
    "probabilities": {
        "no_promotion": 0.15,
        "promotion": 0.85
    },
    "timestamp": "2025-09-24T10:30:00"
}
```

### 4. Batch Prediction
**POST** `/predict_batch`

Predict promotion targeting for multiple users.

**Request:**
```bash
curl -X POST http://localhost:9696/predict_batch \
  -H "Content-Type: application/json" \
  -d '{
    "users": [
        {
            "user_id": "USER-001",
            "age_group": "26-35",
            "location": "Buenos Aires",
            ...
        },
        {
            "user_id": "USER-002",
            "age_group": "18-25",
            "location": "Cordoba",
            ...
        }
    ]
  }'
```

**Response:**
```json
{
    "predictions": [
        {
            "user_id": "USER-001",
            "should_receive_promotion": "Si",
            "confidence": 0.85,
            "probabilities": {...}
        },
        ...
    ],
    "total_users": 2,
    "users_to_target": 1,
    "target_percentage": 50.0,
    "timestamp": "2025-09-24T10:30:00"
}
```

### 5. Synthetic User Prediction
**POST** `/predict_synthetic`

Generate synthetic users and predict their promotion targeting.

**Request:**
```bash
curl -X POST http://localhost:9696/predict_synthetic \
  -H "Content-Type: application/json" \
  -d '{
    "n_samples": 1000,
    "seed": 42
  }'
```

**Response:**
```json
{
    "predictions": [...],
    "total_users": 1000,
    "users_to_target": 423,
    "target_percentage": 42.3,
    "generation_info": {
        "n_samples_requested": 1000,
        "n_samples_generated": 1000,
        "seed": 42
    },
    "timestamp": "2025-09-24T10:30:00"
}
```

### 6. Export Predictions
**POST** `/export_predictions`

Generate predictions and export them to CSV format.

**Request:**
```bash
curl -X POST http://localhost:9696/export_predictions \
  -H "Content-Type: application/json" \
  -d '{
    "n_samples": 100000,
    "include_probabilities": true,
    "format": "csv"
  }'
```

**Response:**
```json
{
    "message": "Predictions exported successfully",
    "file_saved": "predictions/predictions_proba.csv",
    "total_records": 100000,
    "users_to_target": 42350,
    "target_percentage": 42.35,
    "timestamp": "2025-09-24T10:30:00"
}
```

## 📊 Required User Features

All prediction endpoints require the following user features:

| Feature | Type | Description | Example |
|---------|------|-------------|---------|
| `user_id` | string | User identifier (optional for prediction) | "USER-001" |
| `age_group` | string | Age group category | "18-25", "26-35", "36-45", "46-55", "55+" |
| `location` | string | User location | "Buenos Aires", "Cordoba", "Rosario", etc. |
| `device_type` | string | Primary device type | "Mobile", "Desktop", "Tablet" |
| `subscription_type` | string | Subscription level | "Free", "Basic", "Premium", "Enterprise" |
| `days_since_registration` | integer | Days since user registered | 180 |
| `total_purchases` | integer | Total number of purchases | 15 |
| `avg_order_value` | float | Average order value | 150.75 |
| `last_purchase_days` | integer | Days since last purchase | 5 |
| `sessions_last_30_days` | integer | Sessions in last 30 days | 20 |
| `time_on_site_minutes` | float | Average time on site (minutes) | 45.5 |
| `pages_per_session` | float | Average pages viewed per session | 8.2 |
| `cart_abandonment_rate` | float | Cart abandonment rate (0-1) | 0.15 |
| `purchase_frequency` | float | Purchase frequency | 2.5 |

## 🧪 Testing with Python Client

Use the provided client example:

```python
python client_example.py
```

Or create your own client:

```python
import requests

# Single prediction
response = requests.post(
    "http://localhost:9696/predict",
    json={
        "user_id": "USER-001",
        "age_group": "26-35",
        "location": "Buenos Aires",
        # ... other features
    }
)
result = response.json()
print(f"Should receive promotion: {result['should_receive_promotion']}")
```

## 🐳 Docker Deployment

Create a Dockerfile:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements_api.txt .
RUN pip install --no-cache-dir -r requirements_api.txt

COPY . .

EXPOSE 9696

CMD ["python", "api.py"]
```

Build and run:
```bash
docker build -t promotion-api .
docker run -p 9696:9696 promotion-api
```

## 🔍 Error Handling

The API includes comprehensive error handling:

- **400 Bad Request**: Missing or invalid parameters
- **404 Not Found**: Endpoint doesn't exist
- **405 Method Not Allowed**: Wrong HTTP method
- **500 Internal Server Error**: Server-side errors

All error responses follow this format:
```json
{
    "error": "Error type",
    "message": "Detailed error message",
    "details": "Additional information (if available)"
}
```

## 📈 Performance Considerations

- **Batch Size**: For batch predictions, limit to 10,000 users per request
- **Synthetic Generation**: Maximum 10,000 synthetic users per request
- **Export**: For large exports (>100,000 records), consider using batch processing
- **Concurrency**: The Flask development server is single-threaded. For production, use:
  - Gunicorn: `gunicorn -w 4 -b 0.0.0.0:9696 api:app`
  - uWSGI: `uwsgi --http 0.0.0.0:9696 --wsgi-file api.py --callable app`

## 🛠️ Development

### Running in Debug Mode
```python
app.run(debug=True, host='0.0.0.0', port=9696)
```

### Production Configuration
```python
app.run(debug=False, host='0.0.0.0', port=9696, threaded=True)
```

### Environment Variables
```bash
export FLASK_ENV=production
export FLASK_APP=api.py
flask run --host=0.0.0.0 --port=9696
```

## 📝 Logging

The API includes detailed logging for all operations:
- 🔄 Model loading
- 🚀 Incoming requests
- ✅ Successful predictions
- ❌ Errors and exceptions
- 📁 File operations

Logs are output to stdout and can be redirected:
```bash
python api.py > api.log 2>&1
```

## 🔐 Security Notes

For production deployment:
1. Add authentication (API keys, OAuth)
2. Implement rate limiting
3. Use HTTPS with SSL certificates
4. Validate and sanitize all inputs
5. Set appropriate CORS headers
6. Use environment variables for sensitive configuration

## 📚 Additional Resources

- [Flask Documentation](https://flask.palletsprojects.com/)
- [scikit-learn Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [REST API Best Practices](https://restfulapi.net/)
