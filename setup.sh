#!/bin/bash

# MC4 Setup Script
echo "🏭 MC4 Forecasting System Setup"
echo "================================"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

# Check Node.js
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is required but not installed."
    exit 1
fi

echo "✅ Prerequisites check passed"
echo ""

# Backend Setup
echo "📦 Setting up backend..."
cd backend

if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "Generating synthetic data..."
python data_generator.py

echo "Training forecast models..."
python forecast_models.py

echo "Setting up Text2SQL database..."
python setup_chatbot_db.py

echo ""
echo "✅ Backend setup complete!"
echo ""

# Frontend Setup
echo "📦 Setting up frontend..."
cd ../frontend

echo "Installing Node.js dependencies..."
npm install

echo ""
echo "✅ Frontend setup complete!"
echo ""

echo "🎉 Setup complete!"
echo ""
echo "To start the system:"
echo "  1. Backend: cd backend && source venv/bin/activate && python fastapi_server.py"
echo "  2. Frontend: cd frontend && npm run dev"
echo ""
