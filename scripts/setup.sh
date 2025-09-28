#!/bin/bash

echo "🎨 Setting up AI Creative Platform..."

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 18+ first."
    exit 1
fi

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Install frontend dependencies
echo "📦 Installing frontend dependencies..."
cd frontend
npm install
cd ..

# Install backend dependencies
echo "🐍 Installing backend dependencies..."
cd backend

# Check if user has GPU
echo "Do you have an NVIDIA GPU for faster AI processing? (y/n)"
read -r gpu_choice

if [ "$gpu_choice" = "y" ] || [ "$gpu_choice" = "Y" ]; then
    echo "Installing GPU version with CUDA support..."
    pip3 install -r requirements-gpu.txt
else
    echo "Installing CPU-only version..."
    pip3 install -r requirements-cpu.txt
fi

cd ..

# Copy environment files
echo "⚙️ Setting up environment files..."
cp .env.example .env
cp frontend/.env.example frontend/.env
cp backend/.env.example backend/.env

echo "✅ Setup complete!"
echo ""
echo "🚀 To start the development servers:"
echo "   npm run dev"
echo ""
echo "🌐 Your app will be available at:"
echo "   Frontend: http://localhost:5173"
echo "   Backend:  http://localhost:8000"
echo "   API Docs: http://localhost:8000/docs"