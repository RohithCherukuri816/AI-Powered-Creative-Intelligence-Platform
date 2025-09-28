@echo off
echo 🎨 Setting up AI Creative Platform...

REM Check if Node.js is installed
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Node.js is not installed. Please install Node.js 18+ first.
    pause
    exit /b 1
)

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is not installed. Please install Python 3.8+ first.
    pause
    exit /b 1
)

REM Install frontend dependencies
echo 📦 Installing frontend dependencies...
cd frontend
npm install
cd ..

REM Install backend dependencies
echo 🐍 Installing backend dependencies...
cd backend

set /p gpu_choice="Do you have an NVIDIA GPU for faster AI processing? (y/n): "

if /i "%gpu_choice%"=="y" (
    echo Installing GPU version with CUDA support...
    pip install -r requirements-gpu.txt
) else (
    echo Installing CPU-only version...
    pip install -r requirements-cpu.txt
)

cd ..

REM Copy environment files
echo ⚙️ Setting up environment files...
copy .env.example .env
copy frontend\.env.example frontend\.env
copy backend\.env.example backend\.env

echo ✅ Setup complete!
echo.
echo 🚀 To start the development servers:
echo    npm run dev
echo.
echo 🌐 Your app will be available at:
echo    Frontend: http://localhost:5173
echo    Backend:  http://localhost:8000
echo    API Docs: http://localhost:8000/docs

pause