# 🚀 Kaggle Training Guide

## Step-by-Step Instructions to Train on Kaggle and Use Locally

### 📤 Step 1: Upload to Kaggle

1. **Go to Kaggle:**
   - Visit https://www.kaggle.com
   - Sign in or create an account

2. **Create New Notebook:**
   - Click "Code" → "New Notebook"
   - Click "File" → "Upload Notebook"
   - Upload `kaggle_training.ipynb`

3. **Enable GPU:**
   - Click "Settings" (right sidebar)
   - Under "Accelerator", select **GPU T4 x2** or **GPU P100**
   - Click "Save"

### ▶️ Step 2: Run Training

1. **Run All Cells:**
   - Click "Run All" or press `Shift + Enter` on each cell
   - Training will take approximately **2-4 hours** depending on GPU

2. **Monitor Progress:**
   - Watch the training progress in real-time
   - Check accuracy and loss graphs
   - Training will automatically save the best model

### 📥 Step 3: Download Trained Model

1. **After Training Completes:**
   - Look at the right sidebar
   - Click on "Output" tab
   - You'll see `export_for_local` folder

2. **Download Files:**
   - Click the download icon next to `export_for_local`
   - This will download a ZIP file containing:
     - `doodle_classifier.h5` (trained model)
     - `doodle_classifier_metadata.json` (model info)
     - `label_encoder.pkl` (label encoder)
     - `class_names.json` (class names)
     - `training_history.png` (training graphs)

### 💻 Step 4: Use Model Locally

1. **Extract Downloaded Files:**
   ```bash
   # Extract the ZIP file
   # You'll get the export_for_local folder
   ```

2. **Copy to Your Project:**
   ```cmd
   # Windows
   copy export_for_local\doodle_classifier.h5 colab_backend\models\
   copy export_for_local\doodle_classifier_metadata.json colab_backend\models\
   copy export_for_local\label_encoder.pkl colab_backend\models\
   copy export_for_local\class_names.json colab_backend\models\
   ```

   ```bash
   # Linux/Mac
   cp export_for_local/doodle_classifier.h5 colab_backend/models/
   cp export_for_local/doodle_classifier_metadata.json colab_backend/models/
   cp export_for_local/label_encoder.pkl colab_backend/models/
   cp export_for_local/class_names.json colab_backend/models/
   ```

3. **Update Your Recognizer:**
   
   Edit `colab_backend/recognizer.py`:

   ```python
   import os
   import json
   import tensorflow as tf
   import pickle
   
   class DoodleRecognizer:
       def __init__(self):
           model_path = "models/doodle_classifier.h5"
           metadata_path = "models/doodle_classifier_metadata.json"
           
           # Load trained model
           if os.path.exists(model_path):
               print("🔄 Loading trained model...")
               self.model = tf.keras.models.load_model(model_path)
               
               # Load metadata
               with open(metadata_path, 'r') as f:
                   metadata = json.load(f)
               
               self.classes = metadata['class_names']
               self._model_loaded = True
               
               print(f"✅ Loaded trained model with {len(self.classes)} classes")
               print(f"   Test Accuracy: {metadata['test_accuracy']:.4f}")
               print(f"   Top-3 Accuracy: {metadata['test_top3_accuracy']:.4f}")
           else:
               print("⚠️ Trained model not found, using fallback")
               self._model_loaded = False
               # Keep existing fallback code
   ```

4. **Run Your Application:**
   ```bash
   npm run dev
   ```

### 🎯 Expected Results

After training on Kaggle with 100 classes and 10,000 samples each:

- **Training Time:** 2-4 hours on GPU
- **Test Accuracy:** 85-92%
- **Top-3 Accuracy:** 95-98%
- **Model Size:** ~20-50 MB
- **Inference Speed:** <50ms per prediction

### 🔧 Troubleshooting

#### Issue: "Out of Memory" on Kaggle
**Solution:** Reduce `SAMPLES_PER_CATEGORY` in the notebook:
```python
SAMPLES_PER_CATEGORY = 5000  # Instead of 10000
```

#### Issue: "Training Too Slow"
**Solution:** 
- Make sure GPU is enabled in Kaggle settings
- Reduce number of categories
- Reduce epochs

#### Issue: "Can't Load Model Locally"
**Solution:**
```bash
# Install TensorFlow locally
pip install tensorflow==2.13.0
```

#### Issue: "Model Not Found"
**Solution:**
- Make sure you copied files to `colab_backend/models/`
- Check file paths in recognizer.py
- Verify files are not corrupted

### 📊 Customization Options

You can customize the training by editing these variables in the notebook:

```python
# Number of classes to train on
CATEGORIES = ['car', 'cat', 'dog', ...]  # Edit this list

# Samples per category
SAMPLES_PER_CATEGORY = 10000  # Reduce for faster training

# Training parameters
BATCH_SIZE = 128  # Reduce if out of memory
EPOCHS = 30       # Increase for better accuracy
LEARNING_RATE = 0.001  # Adjust learning rate
```

### 🎉 Success Checklist

- [ ] Uploaded notebook to Kaggle
- [ ] Enabled GPU in settings
- [ ] Ran all cells successfully
- [ ] Downloaded trained model files
- [ ] Copied files to local project
- [ ] Updated recognizer.py
- [ ] Tested application locally
- [ ] Model recognizes doodles correctly

### 💡 Pro Tips

1. **Save Kaggle Notebook:** Click "Save Version" regularly to avoid losing progress
2. **Use Quick Test First:** Train on 10 classes first to test the pipeline
3. **Monitor GPU Usage:** Check GPU utilization in Kaggle to ensure it's being used
4. **Download Immediately:** Download files right after training completes
5. **Keep Backup:** Save the Kaggle notebook link for future reference

### 📞 Need Help?

If you encounter issues:
1. Check the error messages in Kaggle notebook
2. Verify GPU is enabled
3. Make sure all files downloaded correctly
4. Check TensorFlow version compatibility

---

**Ready to train?** Upload `kaggle_training.ipynb` to Kaggle and start training! 🚀