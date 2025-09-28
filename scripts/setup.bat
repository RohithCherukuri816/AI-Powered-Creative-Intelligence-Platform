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
pip install -r requirements.txt
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