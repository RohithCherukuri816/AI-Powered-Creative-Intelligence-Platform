#!/usr/bin/env python3
"""
Setup script for Colab backend
Handles all initialization and configuration
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
from typing import Optional, Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ColabSetup:
    """Handles complete setup of the Colab backend environment"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.project_root = self.base_dir.parent
        self.setup_complete = False
        
    def check_environment(self) -> Dict[str, Any]:
        """Check the current environment and capabilities"""
        import torch
        
        env_info = {
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "cpu_count": os.cpu_count(),
            "platform": sys.platform
        }
        
        if torch.cuda.is_available():
            env_info["gpu_name"] = torch.cuda.get_device_name(0)
            env_info["gpu_memory"] = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        logger.info("Environment check completed")
        for key, value in env_info.items():
            logger.info(f"  {key}: {value}")
            
        return env_info
    
    def install_dependencies(self) -> bool:
        """Install all required dependencies"""
        try:
            requirements_file = self.base_dir / "requirements.txt"
            
            if not requirements_file.exists():
                logger.error(f"Requirements file not found: {requirements_file}")
                return False
            
            logger.info("Installing dependencies...")
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", "-q", "-r", str(requirements_file)
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("✅ Dependencies installed successfully")
                return True
            else:
                logger.error(f"❌ Failed to install dependencies: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Exception during dependency installation: {e}")
            return False
    
    def setup_environment_variables(self, **kwargs) -> bool:
        """Setup environment variables"""
        try:
            # Default environment variables
            default_env = {
                "CORS_ALLOWED_ORIGINS": "http://localhost:5173,http://localhost:5174",
                "USE_AI_MODELS": "true",
                "DEVICE": "auto"
            }
            
            # Update with provided kwargs
            default_env.update(kwargs)
            
            # Set environment variables
            for key, value in default_env.items():
                os.environ[key] = str(value)
                logger.info(f"Set {key}={value}")
            
            # Try to get Hugging Face token from Colab secrets
            try:
                from google.colab import userdata
                hf_token = userdata.get('HUGGINGFACE_TOKEN')
                if hf_token:
                    os.environ['HUGGINGFACE_TOKEN'] = hf_token
                    logger.info("✅ Hugging Face token loaded from Colab secrets")
                else:
                    logger.warning("⚠️  Hugging Face token not found in Colab secrets")
            except ImportError:
                logger.info("Not running in Colab, skipping secret loading")
            except Exception as e:
                logger.warning(f"Failed to load Colab secrets: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to setup environment variables: {e}")
            return False
    
    def setup_ngrok(self) -> Optional[str]:
        """Setup ngrok tunnel"""
        try:
            # Install pyngrok if not already installed
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pyngrok"], 
                         capture_output=True)
            
            from pyngrok import ngrok
            from google.colab import userdata
            
            # Get ngrok token
            try:
                ngrok_token = userdata.get('NGROK_AUTH_TOKEN')
                if not ngrok_token:
                    logger.error("❌ NGROK_AUTH_TOKEN not found in Colab secrets")
                    logger.info("Please add your ngrok token to Colab Secrets:")
                    logger.info("1. Go to https://dashboard.ngrok.com/get-started/your-authtoken")
                    logger.info("2. Copy your token")
                    logger.info("3. Add it to Colab Secrets with key 'NGROK_AUTH_TOKEN'")
                    return None
                
                # Authenticate ngrok
                ngrok.set_auth_token(ngrok_token)
                
                # Create tunnel
                public_url = ngrok.connect(8000, 'http')
                logger.info(f"✅ Ngrok tunnel created: {public_url}")
                
                return str(public_url)
                
            except Exception as e:
                logger.error(f"❌ Failed to setup ngrok: {e}")
                return None
                
        except ImportError:
            logger.error("❌ Not running in Colab environment")
            return None
        except Exception as e:
            logger.error(f"❌ Ngrok setup failed: {e}")
            return None
    
    def test_components(self) -> Dict[str, bool]:
        """Test all backend components"""
        results = {}
        
        # Test MobileNet recognizer
        try:
            from .mobilenet_recognizer import MobileNetDoodleRecognizer
            recognizer = MobileNetDoodleRecognizer()
            results["mobilenet_recognizer"] = True
            logger.info("✅ MobileNet recognizer initialized")
        except Exception as e:
            results["mobilenet_recognizer"] = False
            logger.error(f"❌ MobileNet recognizer failed: {e}")
        
        # Test image processor
        try:
            from .processor import ImageProcessor
            processor = ImageProcessor()
            results["image_processor"] = True
            logger.info("✅ Image processor initialized")
        except Exception as e:
            results["image_processor"] = False
            logger.error(f"❌ Image processor failed: {e}")
        
        # Test Hugging Face integration
        try:
            from .huggingface_integration import HuggingFaceIntegration
            hf_integration = HuggingFaceIntegration()
            results["huggingface_integration"] = True
            logger.info("✅ Hugging Face integration initialized")
        except Exception as e:
            results["huggingface_integration"] = False
            logger.error(f"❌ Hugging Face integration failed: {e}")
        
        # Test FastAPI app
        try:
            from .app import create_app
            app = create_app()
            results["fastapi_app"] = True
            logger.info("✅ FastAPI app created")
        except Exception as e:
            results["fastapi_app"] = False
            logger.error(f"❌ FastAPI app failed: {e}")
        
        return results
    
    def start_server(self, host: str = "0.0.0.0", port: int = 8000) -> subprocess.Popen:
        """Start the FastAPI server"""
        try:
            logger.info(f"Starting FastAPI server on {host}:{port}")
            
            proc = subprocess.Popen([
                sys.executable, "-m", "uvicorn", 
                "colab_backend.app:app",
                "--host", host,
                "--port", str(port),
                "--reload"
            ])
            
            logger.info(f"✅ Server started with PID: {proc.pid}")
            return proc
            
        except Exception as e:
            logger.error(f"❌ Failed to start server: {e}")
            raise
    
    def complete_setup(self, **kwargs) -> Dict[str, Any]:
        """Complete setup process"""
        logger.info("🚀 Starting Colab backend setup...")
        
        setup_results = {
            "environment": None,
            "dependencies": False,
            "env_variables": False,
            "ngrok_url": None,
            "component_tests": {},
            "server_process": None,
            "setup_complete": False
        }
        
        # Check environment
        setup_results["environment"] = self.check_environment()
        
        # Install dependencies
        setup_results["dependencies"] = self.install_dependencies()
        if not setup_results["dependencies"]:
            logger.error("❌ Setup failed at dependency installation")
            return setup_results
        
        # Setup environment variables
        setup_results["env_variables"] = self.setup_environment_variables(**kwargs)
        if not setup_results["env_variables"]:
            logger.error("❌ Setup failed at environment variable configuration")
            return setup_results
        
        # Setup ngrok
        setup_results["ngrok_url"] = self.setup_ngrok()
        if not setup_results["ngrok_url"]:
            logger.warning("⚠️  Ngrok setup failed, server will only be accessible locally")
        
        # Test components
        setup_results["component_tests"] = self.test_components()
        
        # Start server
        try:
            setup_results["server_process"] = self.start_server()
            setup_results["setup_complete"] = True
            self.setup_complete = True
            
            logger.info("✅ Colab backend setup completed successfully!")
            
            # Print summary
            self.print_setup_summary(setup_results)
            
        except Exception as e:
            logger.error(f"❌ Failed to start server: {e}")
            setup_results["setup_complete"] = False
        
        return setup_results
    
    def print_setup_summary(self, results: Dict[str, Any]):
        """Print setup summary"""
        print("\n" + "="*60)
        print("🎨 AI-POWERED DOODLE RECOGNITION - COLAB BACKEND")
        print("="*60)
        
        # Environment info
        env = results["environment"]
        print(f"🖥️  Environment:")
        print(f"   Python: {env['python_version'].split()[0]}")
        print(f"   PyTorch: {env['torch_version']}")
        print(f"   CUDA: {'✅ Available' if env['cuda_available'] else '❌ Not available'}")
        if env['cuda_available']:
            print(f"   GPU: {env['gpu_name']} ({env['gpu_memory']:.1f} GB)")
        
        # Component status
        print(f"\n🔧 Components:")
        for component, status in results["component_tests"].items():
            status_icon = "✅" if status else "❌"
            print(f"   {component}: {status_icon}")
        
        # Server info
        print(f"\n🚀 Server:")
        if results["server_process"]:
            print(f"   Status: ✅ Running (PID: {results['server_process'].pid})")
            print(f"   Local: http://localhost:8000")
            if results["ngrok_url"]:
                print(f"   Public: {results['ngrok_url']}")
                print(f"   API Docs: {results['ngrok_url']}/docs")
            print(f"   Health: {results['ngrok_url'] or 'http://localhost:8000'}/health")
        else:
            print(f"   Status: ❌ Failed to start")
        
        # Usage instructions
        print(f"\n📝 Usage Instructions:")
        if results["ngrok_url"]:
            print(f"   1. Copy the public URL: {results['ngrok_url']}")
            print(f"   2. Update your frontend .env: VITE_API_URL={results['ngrok_url']}")
        else:
            print(f"   1. Use local URL: http://localhost:8000")
            print(f"   2. Update your frontend .env: VITE_API_URL=http://localhost:8000")
        print(f"   3. Start your frontend: npm run dev")
        print(f"   4. Draw doodles and generate art!")
        
        print(f"\n⚠️  Note: Keep this Colab session running while using the application")
        print("="*60)

def main():
    """Main setup function"""
    setup = ColabSetup()
    results = setup.complete_setup()
    return results

if __name__ == "__main__":
    main()