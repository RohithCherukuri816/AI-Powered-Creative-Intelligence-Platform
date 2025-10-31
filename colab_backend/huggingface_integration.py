import os
import json
import logging
from typing import Optional, Dict, Any
from huggingface_hub import HfApi, Repository, hf_hub_download, upload_folder
from huggingface_hub.utils import RepositoryNotFoundError
import torch
from pathlib import Path

logger = logging.getLogger(__name__)

class HuggingFaceIntegration:
    """
    Handles all Hugging Face Hub operations for the doodle recognition project
    """
    
    def __init__(self, token: Optional[str] = None):
        self.token = token or os.getenv("HUGGINGFACE_TOKEN")
        self.api = HfApi(token=self.token) if self.token else None
        self.username = None
        
        if self.api:
            try:
                user_info = self.api.whoami()
                self.username = user_info["name"]
                logger.info(f"Authenticated as: {self.username}")
            except Exception as e:
                logger.warning(f"Authentication failed: {e}")
                self.api = None

    def create_model_repository(self, repo_name: str, private: bool = False) -> str:
        """Create a new model repository on Hugging Face Hub"""
        if not self.api:
            raise ValueError("Not authenticated with Hugging Face")
        
        repo_id = f"{self.username}/{repo_name}"
        
        try:
            self.api.create_repo(
                repo_id=repo_id,
                repo_type="model",
                private=private
            )
            logger.info(f"Created model repository: {repo_id}")
            return repo_id
        except Exception as e:
            if "already exists" in str(e):
                logger.info(f"Repository already exists: {repo_id}")
                return repo_id
            else:
                logger.error(f"Failed to create repository: {e}")
                raise

    def create_dataset_repository(self, repo_name: str, private: bool = False) -> str:
        """Create a new dataset repository on Hugging Face Hub"""
        if not self.api:
            raise ValueError("Not authenticated with Hugging Face")
        
        repo_id = f"{self.username}/{repo_name}"
        
        try:
            self.api.create_repo(
                repo_id=repo_id,
                repo_type="dataset",
                private=private
            )
            logger.info(f"Created dataset repository: {repo_id}")
            return repo_id
        except Exception as e:
            if "already exists" in str(e):
                logger.info(f"Repository already exists: {repo_id}")
                return repo_id
            else:
                logger.error(f"Failed to create repository: {e}")
                raise

    def create_space_repository(self, repo_name: str, private: bool = False) -> str:
        """Create a new Space repository on Hugging Face Hub"""
        if not self.api:
            raise ValueError("Not authenticated with Hugging Face")
        
        repo_id = f"{self.username}/{repo_name}"
        
        try:
            self.api.create_repo(
                repo_id=repo_id,
                repo_type="space",
                private=private,
                space_sdk="gradio"
            )
            logger.info(f"Created Space repository: {repo_id}")
            return repo_id
        except Exception as e:
            if "already exists" in str(e):
                logger.info(f"Repository already exists: {repo_id}")
                return repo_id
            else:
                logger.error(f"Failed to create repository: {e}")
                raise

    def upload_model(self, model_dir: str, repo_name: str, commit_message: str = "Upload MobileNet doodle recognition model") -> str:
        """Upload model to Hugging Face Hub"""
        if not self.api:
            raise ValueError("Not authenticated with Hugging Face")
        
        repo_id = f"{self.username}/{repo_name}"
        
        # Ensure repository exists
        try:
            self.api.repo_info(repo_id=repo_id, repo_type="model")
        except RepositoryNotFoundError:
            self.create_model_repository(repo_name)
        
        # Create model card if it doesn't exist
        model_card_path = os.path.join(model_dir, "README.md")
        if not os.path.exists(model_card_path):
            self._create_model_card(model_dir, repo_id)
        
        try:
            # Upload the entire folder
            upload_folder(
                folder_path=model_dir,
                repo_id=repo_id,
                repo_type="model",
                commit_message=commit_message,
                token=self.token
            )
            
            model_url = f"https://huggingface.co/{repo_id}"
            logger.info(f"Model uploaded successfully: {model_url}")
            return model_url
            
        except Exception as e:
            logger.error(f"Failed to upload model: {e}")
            raise

    def upload_dataset(self, dataset_dir: str, repo_name: str, commit_message: str = "Upload Quick Draw processed dataset") -> str:
        """Upload dataset to Hugging Face Hub"""
        if not self.api:
            raise ValueError("Not authenticated with Hugging Face")
        
        repo_id = f"{self.username}/{repo_name}"
        
        # Ensure repository exists
        try:
            self.api.repo_info(repo_id=repo_id, repo_type="dataset")
        except RepositoryNotFoundError:
            self.create_dataset_repository(repo_name)
        
        # Create dataset card if it doesn't exist
        dataset_card_path = os.path.join(dataset_dir, "README.md")
        if not os.path.exists(dataset_card_path):
            self._create_dataset_card(dataset_dir, repo_id)
        
        try:
            # Upload the entire folder
            upload_folder(
                folder_path=dataset_dir,
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=commit_message,
                token=self.token
            )
            
            dataset_url = f"https://huggingface.co/datasets/{repo_id}"
            logger.info(f"Dataset uploaded successfully: {dataset_url}")
            return dataset_url
            
        except Exception as e:
            logger.error(f"Failed to upload dataset: {e}")
            raise

    def create_gradio_space(self, space_dir: str, repo_name: str, commit_message: str = "Create Gradio Space for doodle-to-art demo") -> str:
        """Create and upload Gradio Space"""
        if not self.api:
            raise ValueError("Not authenticated with Hugging Face")
        
        repo_id = f"{self.username}/{repo_name}"
        
        # Ensure repository exists
        try:
            self.api.repo_info(repo_id=repo_id, repo_type="space")
        except RepositoryNotFoundError:
            self.create_space_repository(repo_name)
        
        # Create Space files if they don't exist
        self._create_space_files(space_dir, repo_id)
        
        try:
            # Upload the entire folder
            upload_folder(
                folder_path=space_dir,
                repo_id=repo_id,
                repo_type="space",
                commit_message=commit_message,
                token=self.token
            )
            
            space_url = f"https://huggingface.co/spaces/{repo_id}"
            logger.info(f"Space uploaded successfully: {space_url}")
            return space_url
            
        except Exception as e:
            logger.error(f"Failed to upload Space: {e}")
            raise

    def _create_model_card(self, model_dir: str, repo_id: str):
        """Create a model card for the repository"""
        model_card_content = f"""---
license: mit
tags:
- doodle-recognition
- mobilenet
- image-classification
- quick-draw
- pytorch
language:
- en
pipeline_tag: image-classification
---

# MobileNet Doodle Recognition Model

This model is a MobileNetV2/V3 trained for doodle recognition on the Quick Draw dataset.

## Model Description

- **Architecture**: MobileNetV2/V3
- **Dataset**: Google Quick Draw
- **Categories**: 100 doodle categories
- **Framework**: PyTorch
- **Input Size**: 224x224 RGB images

## Usage

```python
from transformers import AutoModel, AutoConfig
from PIL import Image
import torch

# Load model
model = AutoModel.from_pretrained("{repo_id}")
config = AutoConfig.from_pretrained("{repo_id}")

# Process image
image = Image.open("your_doodle.png")
# Add your preprocessing here

# Inference
with torch.no_grad():
    outputs = model(processed_image)
    predictions = torch.softmax(outputs, dim=1)
```

## Training

This model was trained using transfer learning from ImageNet pre-trained weights.

## Performance

- Efficient inference suitable for mobile deployment
- Optimized for real-time doodle recognition
- Lightweight architecture for web applications

## Categories

The model recognizes 100 categories including: airplane, car, cat, dog, house, tree, and many more.

## License

MIT License - see LICENSE file for details.
"""
        
        with open(os.path.join(model_dir, "README.md"), "w") as f:
            f.write(model_card_content)

    def _create_dataset_card(self, dataset_dir: str, repo_id: str):
        """Create a dataset card for the repository"""
        dataset_card_content = f"""---
license: cc-by-4.0
tags:
- doodle
- quick-draw
- image-classification
- computer-vision
task_categories:
- image-classification
language:
- en
size_categories:
- 100K<n<1M
---

# Quick Draw Processed Dataset

This dataset contains processed doodles from Google's Quick Draw dataset, prepared for training doodle recognition models.

## Dataset Description

- **Source**: Google Quick Draw Dataset
- **Categories**: 100 doodle categories
- **Format**: Processed numpy arrays and images
- **Size**: Varies by category
- **License**: Creative Commons Attribution 4.0

## Dataset Structure

```
dataset/
├── train/
│   ├── category1/
│   ├── category2/
│   └── ...
├── test/
│   ├── category1/
│   ├── category2/
│   └── ...
└── metadata.json
```

## Usage

```python
from datasets import load_dataset

# Load dataset
dataset = load_dataset("{repo_id}")

# Access training data
train_data = dataset['train']
test_data = dataset['test']
```

## Categories

The dataset includes 100 categories: airplane, ambulance, angel, ant, apple, axe, banana, baseball, basketball, bat, and many more.

## Preprocessing

- Images are normalized to 28x28 grayscale
- Data augmentation applied during training
- Balanced sampling across categories

## Citation

```
@misc{{quickdraw-processed,
  title={{Quick Draw Processed Dataset}},
  author={{{repo_id.split('/')[0]}}},
  year={{2025}},
  publisher={{Hugging Face}},
  url={{https://huggingface.co/datasets/{repo_id}}}
}}
```
"""
        
        with open(os.path.join(dataset_dir, "README.md"), "w") as f:
            f.write(dataset_card_content)

    def _create_space_files(self, space_dir: str, repo_id: str):
        """Create necessary files for Gradio Space"""
        os.makedirs(space_dir, exist_ok=True)
        
        # Create app.py for Gradio Space
        app_py_content = '''import gradio as gr
import torch
from PIL import Image
import numpy as np
from transformers import AutoModel, AutoConfig

# Load model
model_repo = "vinayabc1824/doodle-recognition-mobilenet"
try:
    model = AutoModel.from_pretrained(model_repo)
    config = AutoConfig.from_pretrained(model_repo)
    model.eval()
    print("✅ Model loaded successfully")
except:
    model = None
    print("❌ Model loading failed")

def recognize_doodle(image):
    """Recognize doodle from image"""
    if model is None:
        return "Model not available", 0.0
    
    try:
        # Preprocess image
        if image is None:
            return "No image provided", 0.0
        
        # Convert to RGB and resize
        image = image.convert('RGB')
        image = image.resize((224, 224))
        
        # Convert to tensor (simplified preprocessing)
        image_array = np.array(image)
        image_tensor = torch.FloatTensor(image_array).permute(2, 0, 1).unsqueeze(0) / 255.0
        
        # Inference
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            
            # Get class name (you'd need to load the actual class names)
            classes = config.classes if hasattr(config, 'classes') else [f"class_{i}" for i in range(100)]
            predicted_class = classes[predicted_idx.item()]
            confidence_score = confidence.item()
            
            return predicted_class, confidence_score
    
    except Exception as e:
        return f"Error: {str(e)}", 0.0

# Create Gradio interface
with gr.Blocks(title="🎨 Doodle Recognition") as demo:
    gr.Markdown("# 🎨 AI-Powered Doodle Recognition")
    gr.Markdown("Draw a doodle and let our MobileNet model recognize what it is!")
    
    with gr.Row():
        with gr.Column():
            canvas = gr.Sketchpad(
                label="Draw your doodle here",
                type="pil",
                image_mode="RGB",
                canvas_size=(280, 280),
                brush_radius=3
            )
            recognize_btn = gr.Button("🔍 Recognize Doodle", variant="primary")
        
        with gr.Column():
            result_label = gr.Textbox(label="Recognized Object", interactive=False)
            confidence_score = gr.Number(label="Confidence Score", interactive=False)
    
    # Examples
    gr.Markdown("### Try these examples:")
    gr.Examples(
        examples=[
            ["Draw a car"],
            ["Draw a cat"], 
            ["Draw a house"],
            ["Draw a tree"],
            ["Draw a flower"]
        ],
        inputs=[gr.Textbox(visible=False)],
        label="Example prompts"
    )
    
    recognize_btn.click(
        fn=recognize_doodle,
        inputs=[canvas],
        outputs=[result_label, confidence_score]
    )

if __name__ == "__main__":
    demo.launch()
'''
        
        with open(os.path.join(space_dir, "app.py"), "w") as f:
            f.write(app_py_content)
        
        # Create requirements.txt
        requirements_content = """gradio==4.0.0
torch==2.0.0
torchvision==0.15.0
transformers==4.35.0
Pillow==10.0.0
numpy==1.24.0
"""
        
        with open(os.path.join(space_dir, "requirements.txt"), "w") as f:
            f.write(requirements_content)
        
        # Create README.md for Space
        readme_content = f"""---
title: Doodle Recognition Demo
emoji: 🎨
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: mit
---

# 🎨 Doodle Recognition Demo

An interactive demo of MobileNet-based doodle recognition trained on the Quick Draw dataset.

## Features

- 🖊️ Interactive drawing canvas
- 🤖 Real-time doodle recognition
- 📊 Confidence scoring
- 🎯 100+ category support

## Usage

1. Draw a doodle in the canvas
2. Click "Recognize Doodle"
3. See the AI's prediction and confidence score

## Model

This demo uses a MobileNetV2/V3 model trained on Google's Quick Draw dataset, achieving efficient recognition of hand-drawn sketches.
"""
        
        with open(os.path.join(space_dir, "README.md"), "w") as f:
            f.write(readme_content)

    def get_repository_info(self, repo_id: str, repo_type: str = "model") -> Dict[str, Any]:
        """Get information about a repository"""
        if not self.api:
            raise ValueError("Not authenticated with Hugging Face")
        
        try:
            info = self.api.repo_info(repo_id=repo_id, repo_type=repo_type)
            return {
                "id": info.id,
                "author": info.author,
                "sha": info.sha,
                "created_at": info.created_at,
                "last_modified": info.last_modified,
                "private": info.private,
                "downloads": getattr(info, 'downloads', 0),
                "likes": getattr(info, 'likes', 0)
            }
        except Exception as e:
            logger.error(f"Failed to get repository info: {e}")
            raise

    def list_user_repositories(self, repo_type: str = "model") -> list:
        """List all repositories for the authenticated user"""
        if not self.api:
            raise ValueError("Not authenticated with Hugging Face")
        
        try:
            repos = self.api.list_repos(author=self.username, repo_type=repo_type)
            return [{"id": repo.id, "private": repo.private, "last_modified": repo.last_modified} for repo in repos]
        except Exception as e:
            logger.error(f"Failed to list repositories: {e}")
            raise