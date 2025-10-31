# 🎨 Colab Backend - AI-Powered Doodle Recognition

Complete backend implementation for running the AI-Powered Doodle Recognition platform on Google Colab with GPU acceleration.

## 🚀 Features

- **🤖 MobileNetV2/V3**: Efficient doodle recognition optimized for mobile deployment
- **🎨 Stable Diffusion**: High-quality image generation with ControlNet structure preservation
- **⚡ GPU Acceleration**: Optimized for Google Colab's T4 GPUs
- **🤗 Hugging Face Ready**: Built-in integration for model upload and sharing
- **📊 Training Pipeline**: Complete training workflow for custom models
- **🌐 Public Access**: Ngrok tunneling for external frontend connections

## 📦 Quick Start

### Option 1: Use the Colab Notebook (Recommended)

1. Open `Colab_Backend.ipynb` in Google Colab
2. Set your `NGROK_AUTH_TOKEN` in Colab Secrets
3. Run all cells in order
4. Copy the public URL for your frontend

### Option 2: Manual Setup

```python
from colab_backend.setup_colab import ColabSetup

# Complete automated setup
setup = ColabSetup()
results = setup.complete_setup()

# Get the public URL
public_url = results["ngrok_url"]
print(f"Backend running at: {public_url}")
```

## 🔧 Components

### MobileNet Recognizer
```python
from colab_backend.mobilenet_recognizer import MobileNetDoodleRecognizer

recognizer = MobileNetDoodleRecognizer()
label, confidence = recognizer.recognize_doodle(image)
```

### Model Training
```python
from colab_backend.mobilenet_trainer import MobileNetTrainer

trainer = MobileNetTrainer()
trainer.create_model(pretrained=True)
trainer.train(epochs=10)
```

### Hugging Face Integration
```python
from colab_backend.huggingface_integration import HuggingFaceIntegration

hf = HuggingFaceIntegration()
model_url = hf.upload_model("model_dir", "doodle-recognition-mobilenet")
```

## 📋 Requirements

- Google Colab with GPU runtime
- Ngrok account and auth token
- Hugging Face account (optional, for model upload)

## 🛠️ Configuration

### Environment Variables
- `HUGGINGFACE_TOKEN`: Your Hugging Face access token
- `NGROK_AUTH_TOKEN`: Your ngrok authentication token (in Colab Secrets)
- `CORS_ALLOWED_ORIGINS`: Frontend URLs (automatically configured)

### Colab Secrets Setup
1. Go to Colab → Secrets (🔑 icon)
2. Add `NGROK_AUTH_TOKEN` with your token from [ngrok dashboard](https://dashboard.ngrok.com/get-started/your-authtoken)
3. Optionally add `HUGGINGFACE_TOKEN` for model uploads

## 🎯 API Endpoints

Once running, the backend provides:

- `GET /`: Server information
- `GET /health`: Health check with GPU status
- `POST /generate-design`: Main doodle-to-art endpoint
- `GET /supported-labels`: List of recognized categories
- `GET /model-info`: Current model information

## 📊 Training Your Own Model

```python
# 1. Initialize trainer
trainer = MobileNetTrainer(num_classes=10)

# 2. Download data
categories = ['car', 'cat', 'dog', 'house', 'tree']
images, labels, classes = trainer.download_quickdraw_data(categories)

# 3. Prepare data
trainer.prepare_data(images, labels)

# 4. Create and train model
trainer.create_model(pretrained=True)
trainer.train(epochs=10)

# 5. Save for Hugging Face
model_dir = trainer.save_model_for_huggingface()
```

## 🤗 Uploading to Hugging Face

```python
from colab_backend.huggingface_integration import HuggingFaceIntegration

hf = HuggingFaceIntegration()

# Upload model
model_url = hf.upload_model("model_dir", "my-doodle-model")

# Create Gradio Space
space_url = hf.create_gradio_space("space_dir", "doodle-demo")
```

## 🔍 Monitoring

The setup provides real-time monitoring of:
- GPU memory usage
- Server status
- Model performance
- API request logs

## 🐛 Troubleshooting

### Common Issues

1. **Ngrok token error**: Make sure `NGROK_AUTH_TOKEN` is set in Colab Secrets
2. **GPU not available**: Ensure Colab runtime is set to GPU (T4)
3. **Model loading fails**: Check internet connection and Hugging Face token
4. **CORS errors**: Verify frontend URL is in `CORS_ALLOWED_ORIGINS`

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run setup with debug info
setup = ColabSetup()
results = setup.complete_setup()
```

## 📈 Performance

- **Model Size**: ~25MB (MobileNetV3)
- **Inference Time**: <50ms on T4 GPU
- **Training Time**: ~30 minutes for 10 classes
- **Memory Usage**: ~2GB GPU memory

## 🔄 Updates

The backend automatically handles:
- Model version management
- Dependency updates
- Configuration changes
- Error recovery

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Test in Colab environment
4. Submit a pull request

## 📞 Support

For issues and questions:
- Check the troubleshooting section
- Review Colab notebook outputs
- Open an issue on GitHub