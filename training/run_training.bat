@echo off
REM Complete training pipeline script for GPU machine (Windows)
REM This script will run the entire training process from start to finish

echo 🎨 Doodle Recognition Model Training Pipeline
echo ==============================================

REM Check if we're in the training directory
if not exist "config.py" (
    echo ❌ Please run this script from the training directory
    pause
    exit /b 1
)

REM Check Python and dependencies
echo 🔍 Checking dependencies...
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')" >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ TensorFlow not found. Please install requirements first:
    echo    pip install -r requirements.txt
    pause
    exit /b 1
)

REM Check GPU availability
echo 🔍 Checking GPU availability...
python -c "import tensorflow as tf; print(f'GPU Available: {tf.config.list_physical_devices(\"GPU\")}')"

echo.
echo Choose training mode:
echo 1. Quick training (10 classes, 1000 samples each, ~30 minutes)
echo 2. Full training (100 classes, 10000 samples each, ~4 hours)
echo 3. Custom training (specify parameters)
echo.

set /p choice="Enter choice (1-3): "

if "%choice%"=="1" (
    echo ⚡ Starting quick training...
    python quick_train.py
    goto :end
)

if "%choice%"=="2" (
    echo 🚀 Starting full training...
    
    REM Download dataset
    echo 📥 Step 1: Downloading dataset...
    python download_dataset.py
    if %errorlevel% neq 0 (
        echo ❌ Dataset download failed
        pause
        exit /b 1
    )
    
    REM Preprocess data
    echo 🔄 Step 2: Preprocessing data...
    python data_preprocessing.py
    if %errorlevel% neq 0 (
        echo ❌ Data preprocessing failed
        pause
        exit /b 1
    )
    
    REM Train model
    echo 🏋️ Step 3: Training model...
    python train_model.py --architecture simple
    if %errorlevel% neq 0 (
        echo ❌ Model training failed
        pause
        exit /b 1
    )
    
    REM Evaluate model
    echo 📊 Step 4: Evaluating model...
    python evaluate_model.py --model models/doodle_classifier_simple_final.h5
    if %errorlevel% neq 0 (
        echo ❌ Model evaluation failed
        pause
        exit /b 1
    )
    
    REM Export model
    echo 📦 Step 5: Exporting model...
    python export_model.py --model models/doodle_classifier_simple_final.h5
    if %errorlevel% neq 0 (
        echo ❌ Model export failed
        pause
        exit /b 1
    )
    
    goto :end
)

if "%choice%"=="3" (
    echo ⚙️ Custom training...
    
    REM Ask for parameters
    set /p arch="Architecture (simple/advanced/lightweight): "
    set /p epochs="Number of epochs (default 50): "
    set /p batch_size="Batch size (default 128): "
    
    REM Set defaults
    if "%arch%"=="" set arch=simple
    if "%epochs%"=="" set epochs=50
    if "%batch_size%"=="" set batch_size=128
    
    echo 🔄 Starting custom training with:
    echo    Architecture: %arch%
    echo    Epochs: %epochs%
    echo    Batch size: %batch_size%
    
    REM Download and preprocess if needed
    if not exist "data\processed" (
        echo 📥 Downloading and preprocessing data...
        python download_dataset.py
        python data_preprocessing.py
    )
    
    REM Train with custom parameters
    python train_model.py --architecture %arch%
    
    REM Evaluate
    python evaluate_model.py --model models/doodle_classifier_%arch%_final.h5
    
    REM Export
    python export_model.py --model models/doodle_classifier_%arch%_final.h5
    
    goto :end
)

echo ❌ Invalid choice
pause
exit /b 1

:end
echo.
echo 🎉 Training pipeline completed!
echo.
echo 📁 Generated files:
echo    - models\          : Trained model files
echo    - plots\           : Training visualizations
echo    - exported_models\ : Production-ready model
echo    - logs\            : Training logs
echo.
echo 🚀 Next steps:
echo    1. Copy the exported model to your main project
echo    2. Update your recognizer to use the trained model
echo    3. Test the model with real doodles
echo.
echo 📋 To copy to main project:
echo    copy exported_models\* ..\colab_backend\models\
echo.
pause