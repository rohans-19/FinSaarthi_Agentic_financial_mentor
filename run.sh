#!/bin/bash
# Start FastAPI backend
echo "Starting FastAPI backend on port 8000..."
uvicorn api:app --reload --port 8000 &
BACKEND_PID=$!

# Start React frontend
echo "Starting React frontend..."
if [ -d "frontend" ]; then
    cd frontend
    if [ ! -d "node_modules" ]; then
        echo "node_modules not found. Installing..."
        npm install
    fi
    npm run dev
else
    echo "Error: frontend directory not found."
fi

# Cleanup on exit
trap "kill $BACKEND_PID" EXIT
