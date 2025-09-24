#!/bin/bash

# User Promotion Targeting Prediction API - Runner Script
# This script helps run and test the Flask API

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
API_PORT=9696
API_HOST="localhost"
BASE_URL="http://${API_HOST}:${API_PORT}"

# Function to print colored output
print_color() {
    color=$1
    message=$2
    echo -e "${color}${message}${NC}"
}

# Function to check if API is running
check_api_status() {
    if curl -s -o /dev/null -w "%{http_code}" "${BASE_URL}/health" | grep -q "200"; then
        return 0
    else
        return 1
    fi
}

# Function to install dependencies
install_deps() {
    print_color "$BLUE" "\n📦 Installing dependencies..."
    pip install -r requirements_api.txt
    print_color "$GREEN" "✅ Dependencies installed successfully"
}

# Function to start the API
start_api() {
    print_color "$BLUE" "\n🚀 Starting Flask API on port ${API_PORT}..."
    
    # Check if port is already in use
    if lsof -Pi :${API_PORT} -sTCP:LISTEN -t >/dev/null ; then
        print_color "$RED" "❌ Port ${API_PORT} is already in use!"
        print_color "$YELLOW" "Stop the existing service or use a different port."
        exit 1
    fi
    
    # Start the API in background
    python api.py &
    API_PID=$!
    print_color "$GREEN" "✅ API started with PID: ${API_PID}"
    
    # Wait for API to be ready
    print_color "$YELLOW" "⏳ Waiting for API to be ready..."
    sleep 3
    
    if check_api_status; then
        print_color "$GREEN" "✅ API is running and healthy!"
    else
        print_color "$RED" "❌ API failed to start properly"
        exit 1
    fi
}

# Function to stop the API
stop_api() {
    print_color "$BLUE" "\n🛑 Stopping Flask API..."
    
    # Find and kill the API process
    API_PID=$(lsof -ti:${API_PORT})
    if [ ! -z "$API_PID" ]; then
        kill $API_PID
        print_color "$GREEN" "✅ API stopped (PID: ${API_PID})"
    else
        print_color "$YELLOW" "⚠️ No API process found on port ${API_PORT}"
    fi
}

# Function to test the API
test_api() {
    print_color "$BLUE" "\n🧪 Testing API endpoints..."
    
    # Check if API is running
    if ! check_api_status; then
        print_color "$RED" "❌ API is not running. Start it first with: ./run_api.sh start"
        exit 1
    fi
    
    # Test 1: Health Check
    print_color "$YELLOW" "\n1️⃣ Testing health endpoint..."
    curl -s "${BASE_URL}/health" | python -m json.tool
    
    # Test 2: Model Info
    print_color "$YELLOW" "\n2️⃣ Testing model info endpoint..."
    curl -s "${BASE_URL}/model_info" | python -m json.tool | head -20
    
    # Test 3: Single Prediction
    print_color "$YELLOW" "\n3️⃣ Testing single prediction..."
    curl -s -X POST "${BASE_URL}/predict" \
        -H "Content-Type: application/json" \
        -d '{
            "user_id": "TEST-USER-001",
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
        }' | python -m json.tool
    
    # Test 4: Synthetic Prediction
    print_color "$YELLOW" "\n4️⃣ Testing synthetic prediction (10 samples)..."
    curl -s -X POST "${BASE_URL}/predict_synthetic" \
        -H "Content-Type: application/json" \
        -d '{"n_samples": 10, "seed": 42}' | python -m json.tool | head -50
    
    print_color "$GREEN" "\n✅ All tests completed successfully!"
}

# Function to run the client example
run_client() {
    print_color "$BLUE" "\n🔧 Running client example..."
    
    # Check if API is running
    if ! check_api_status; then
        print_color "$RED" "❌ API is not running. Start it first with: ./run_api.sh start"
        exit 1
    fi
    
    python client_example.py
}

# Function to show logs
show_logs() {
    print_color "$BLUE" "\n📋 Showing API logs..."
    
    # Find the API process
    API_PID=$(lsof -ti:${API_PORT})
    if [ ! -z "$API_PID" ]; then
        print_color "$YELLOW" "Showing logs for PID: ${API_PID} (Press Ctrl+C to stop)"
        # Note: This will show future logs only
        tail -f /tmp/api.log 2>/dev/null || print_color "$YELLOW" "No log file found. Run API with logging enabled."
    else
        print_color "$YELLOW" "⚠️ No API process found on port ${API_PORT}"
    fi
}

# Function to show usage
show_usage() {
    echo "User Promotion Targeting Prediction API - Runner Script"
    echo ""
    echo "Usage: $0 {install|start|stop|restart|status|test|client|logs|help}"
    echo ""
    echo "Commands:"
    echo "  install  - Install required dependencies"
    echo "  start    - Start the Flask API server"
    echo "  stop     - Stop the Flask API server"
    echo "  restart  - Restart the Flask API server"
    echo "  status   - Check if API is running"
    echo "  test     - Run basic API tests"
    echo "  client   - Run the Python client example"
    echo "  logs     - Show API logs"
    echo "  help     - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 install   # Install dependencies"
    echo "  $0 start     # Start the API"
    echo "  $0 test      # Test API endpoints"
    echo "  $0 client    # Run client example"
}

# Main script logic
case "$1" in
    install)
        install_deps
        ;;
    start)
        start_api
        ;;
    stop)
        stop_api
        ;;
    restart)
        stop_api
        sleep 2
        start_api
        ;;
    status)
        if check_api_status; then
            print_color "$GREEN" "✅ API is running and healthy"
            curl -s "${BASE_URL}/health" | python -m json.tool
        else
            print_color "$RED" "❌ API is not running or not responding"
        fi
        ;;
    test)
        test_api
        ;;
    client)
        run_client
        ;;
    logs)
        show_logs
        ;;
    help|--help|-h)
        show_usage
        ;;
    *)
        print_color "$RED" "Invalid command: $1"
        echo ""
        show_usage
        exit 1
        ;;
esac
