from PIL import Image, ImageDraw, ImageFilter, ImageEnhance, ImageOps
import torch
import numpy as np
from typing import Tuple, Optional
import cv2
import os
import logging
import random
from config.ai_config import get_ai_config, STYLE_ENHANCEMENTS, QUALITY_ENHANCERS, NEGATIVE_PROMPTS

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageProcessor:
    """
    AI-powered image processing using ControlNet + Stable Diffusion.
    Transforms sketches into stunning artwork while preserving structure.
    Singleton pattern to ensure models are loaded only once.
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ImageProcessor, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        # Only initialize once
        if ImageProcessor._initialized:
            return
            
        # Load configuration
        self.config = get_ai_config()
        
        # Set device
        if self.config["device"] == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = self.config["device"]
        
        logger.info(f"Using device: {self.device}")
        logger.info(f"AI models enabled: {self.config['use_ai_models']}")
        
        # Initialize ControlNet models (lazy loading)
        self._ai_modules_loaded = False
        self._models_loaded = False  # Add explicit flag for model loading state
        self.controlnet_canny = None
        self.pipe_canny = None
        
        # Preprocessors (lazy loading)
        self.canny_detector = None
        
        # Cache for AI modules to prevent reimporting
        self._StableDiffusionControlNetPipeline = None
        self._ControlNetModel = None
        self._CannyDetector = None
        
        # Mark as initialized
        ImageProcessor._initialized = True
        
        # Fallback colors for mock mode
        self.pastel_colors = [
            (230, 230, 250),  # Lavender
            (255, 182, 193),  # Light Pink
            (176, 224, 230),  # Powder Blue
            (240, 255, 240),  # Honeydew
            (255, 218, 185),  # Peach Puff
            (221, 160, 221),  # Plum
            (255, 255, 224),  # Light Yellow
        ]
    
    def are_models_loaded(self):
        """Check if AI models are loaded without triggering a load"""
        return self._models_loaded and self.controlnet_canny is not None and self.pipe_canny is not None
    
    def apply_style_transformation(self, image: Image.Image, prompt: str, generation_id: str) -> Image.Image:
        """
        Apply AI-powered style transformation using ControlNet + Stable Diffusion.
        Falls back to traditional processing if AI models aren't available.
        """
        try:
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize to optimal size for ControlNet (512x512 or 768x768)
            target_size = 512
            image = self._resize_for_controlnet(image, target_size)
            
            # Try AI-powered generation first
            result = self._generate_with_controlnet(image, prompt)
            if result is not None:
                logger.info(f"Successfully generated image with ControlNet for {generation_id}")
                return result
            
            # Fallback to traditional processing
            logger.warning("ControlNet not available, using fallback processing")
            return self._apply_fallback_processing(image, prompt)
            
        except Exception as e:
            logger.error(f"Error in style transformation: {str(e)}")
            # Return processed version as fallback
            return self._apply_fallback_processing(image, prompt)
    
    def _load_controlnet_models(self):
        """Lazy load ControlNet models to save memory and avoid heavy imports when disabled"""
        try:
            # Check if AI models are enabled
            if not self.config["use_ai_models"]:
                logger.info("AI models disabled in configuration")
                return False

            # Check if models are already loaded and cached properly
            if (self._models_loaded and 
                self.controlnet_canny is not None and 
                self.pipe_canny is not None and
                self.canny_detector is not None):
                logger.info("✅ Models already loaded and cached, reusing...")
                return True

            # Import heavy modules lazily (only once)
            if not self._ai_modules_loaded:
                logger.info("Importing AI modules (diffusers/controlnet-aux)...")
                try:
                    from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
                    from controlnet_aux import CannyDetector
                    
                    # Cache the classes to prevent reimporting
                    self._StableDiffusionControlNetPipeline = StableDiffusionControlNetPipeline
                    self._ControlNetModel = ControlNetModel
                    self._CannyDetector = CannyDetector
                    self._ai_modules_loaded = True
                    logger.info("✅ AI modules imported and cached")
                except Exception as e:
                    logger.error(f"Failed to import AI modules: {e}")
                    return False

            # Load models only if not already loaded
            if not self._models_loaded:
                logger.info("🔄 Loading ControlNet models and pipeline (this will take a moment)...")

                # Load preprocessors
                if self.canny_detector is None:
                    logger.info("Loading Canny detector...")
                    self.canny_detector = self._CannyDetector()

                # Load ControlNet model
                auth_token = self.config.get("huggingface_token")
                dtype = torch.float16 if self.device == "cuda" else torch.float32
                
                logger.info("Loading ControlNet model...")
                self.controlnet_canny = self._ControlNetModel.from_pretrained(
                    self.config["controlnet_model"],
                    torch_dtype=dtype,
                    use_safetensors=True,
                    token=auth_token
                )

                # Load Stable Diffusion pipeline
                logger.info("Loading Stable Diffusion pipeline...")
                self.pipe_canny = self._StableDiffusionControlNetPipeline.from_pretrained(
                    self.config["stable_diffusion_model"],
                    controlnet=self.controlnet_canny,
                    torch_dtype=dtype,
                    safety_checker=None,
                    requires_safety_checker=False,
                    use_safetensors=True,
                    token=auth_token
                )
                
                logger.info(f"Moving pipeline to device: {self.device}")
                self.pipe_canny.to(self.device)

                # Enable memory optimizations
                if hasattr(self.pipe_canny, "enable_xformers_memory_efficient_attention"):
                    try:
                        self.pipe_canny.enable_xformers_memory_efficient_attention()
                        logger.info("XFormers memory optimization enabled")
                    except Exception as e:
                        logger.warning(f"Could not enable XFormers: {e}")

                # Enable CPU offloading for low VRAM systems
                if self.device == "cuda" and hasattr(self.pipe_canny, "enable_model_cpu_offload"):
                    try:
                        self.pipe_canny.enable_model_cpu_offload()
                        logger.info("Model CPU offloading enabled")
                    except Exception as e:
                        logger.warning(f"Could not enable CPU offloading: {e}")

                # Mark as loaded to prevent reloading
                self._models_loaded = True
                logger.info("✅ ControlNet models loaded successfully and cached!")
                return True

            return True

        except Exception as e:
            logger.error(f"Failed to load ControlNet models: {str(e)}")
            logger.info("Falling back to traditional image processing")
            return False
    
    def _resize_for_controlnet(self, image: Image.Image, target_size: int = 512) -> Image.Image:
        """Resize image to optimal size for ControlNet while maintaining aspect ratio"""
        width, height = image.size
        
        # Calculate new dimensions maintaining aspect ratio
        if width > height:
            new_width = target_size
            new_height = int((height * target_size) / width)
        else:
            new_height = target_size
            new_width = int((width * target_size) / height)
        
        # Ensure dimensions are multiples of 8 (required by Stable Diffusion)
        new_width = (new_width // 8) * 8
        new_height = (new_height // 8) * 8
        
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    def _generate_with_controlnet(self, image: Image.Image, prompt: str) -> Optional[Image.Image]:
        """Generate image using ControlNet + Stable Diffusion"""
        try:
            # Ensure models are loaded (this should be very fast after first load)
            if not self._models_loaded or self.controlnet_canny is None or self.pipe_canny is None:
                logger.info("Models not loaded, loading now...")
                if not self._load_controlnet_models():
                    return None
            else:
                logger.info("✅ Using cached models for generation")
            
            # Enhance the prompt for better results
            enhanced_prompt = self._enhance_prompt(prompt)
            negative_prompt = self._build_negative_prompt()

            params = self.config.get("generation_params", {})
            
            # Detect control method based on image content
            control_method = self._detect_best_control_method(image)
            
            if control_method == "canny":
                # Use Canny edge detection
                control_image = self.canny_detector(image)
                pipe = self.pipe_canny
            else:
                # Fallback to canny if lineart fails
                control_image = self.canny_detector(image)
                pipe = self.pipe_canny
            
            # Generate image
            logger.info(f"Generating with prompt: {enhanced_prompt}")
            
            # Use CPU generator by default to avoid device issues
            if self.device == "cuda" and torch.cuda.is_available():
                generator = torch.Generator(device="cuda").manual_seed(42)
            else:
                generator = torch.Generator().manual_seed(42)

            result = pipe(
                prompt=enhanced_prompt,
                negative_prompt=negative_prompt,
                image=control_image,
                num_inference_steps=int(params.get("num_inference_steps", 20)),
                guidance_scale=float(params.get("guidance_scale", 7.5)),
                controlnet_conditioning_scale=float(params.get("controlnet_conditioning_scale", 1.0)),
                generator=generator
            ).images[0]
            
            return result
            
        except Exception as e:
            logger.error(f"ControlNet generation failed: {str(e)}")
            return None
    
    def _enhance_prompt(self, prompt: str) -> str:
        """Enhance the user prompt for better AI generation"""
        # Add quality and style modifiers
        quality_terms = ", ".join(QUALITY_ENHANCERS)
        
        # Detect style and add appropriate modifiers
        style_modifiers = ""
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ["watercolor", "painting"]):
            style_modifiers = "watercolor painting, soft brushstrokes, artistic"
        elif any(word in prompt_lower for word in ["digital", "cyber", "neon"]):
            style_modifiers = "digital art, vibrant colors, modern"
        elif any(word in prompt_lower for word in ["minimalist", "simple"]):
            style_modifiers = "minimalist, clean lines, simple"
        elif any(word in prompt_lower for word in ["vintage", "retro"]):
            style_modifiers = "vintage style, retro, classic"
        elif any(word in prompt_lower for word in ["anime", "manga"]):
            style_modifiers = "anime style, manga, japanese art"
        
        # Combine all parts
        enhanced = f"{prompt}, {style_modifiers}, {quality_terms}" if style_modifiers else f"{prompt}, {quality_terms}"
        
        return enhanced

    def _build_negative_prompt(self) -> str:
        """Build a negative prompt string from configuration"""
        try:
            return ", ".join(NEGATIVE_PROMPTS)
        except Exception:
            return ""
    
    def _detect_best_control_method(self, image: Image.Image) -> str:
        """Detect the best ControlNet method based on image characteristics"""
        # Convert to numpy array for analysis
        img_array = np.array(image)
        
        # Calculate edge density using Canny
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # If image has clear edges/lines, use canny
        if edge_density > 0.05:  # Threshold for edge density
            return "canny"
        else:
            return "canny"  # Default to canny for now
    
    def _apply_fallback_processing(self, image: Image.Image, prompt: str) -> Image.Image:
        """Fallback to traditional image processing when AI models aren't available"""
        logger.info("Using fallback image processing")
        
        # Determine style based on prompt keywords
        style = self._detect_style_from_prompt(prompt.lower())
        
        # Apply the appropriate transformation
        if style == "watercolor":
            return self._apply_watercolor_effect(image)
        elif style == "digital_art":
            return self._apply_digital_art_effect(image)
        elif style == "minimalist":
            return self._apply_minimalist_effect(image)
        elif style == "vintage":
            return self._apply_vintage_effect(image)
        elif style == "geometric":
            return self._apply_geometric_effect(image)
        else:
            return self._apply_general_artistic_effect(image)
    
    def _detect_style_from_prompt(self, prompt: str) -> str:
        """Detect the intended style from the prompt text"""
        if any(word in prompt for word in ["watercolor", "painting", "soft", "dreamy"]):
            return "watercolor"
        elif any(word in prompt for word in ["digital", "vibrant", "neon", "cyber"]):
            return "digital_art"
        elif any(word in prompt for word in ["minimalist", "simple", "clean", "line"]):
            return "minimalist"
        elif any(word in prompt for word in ["vintage", "retro", "old", "classic"]):
            return "vintage"
        elif any(word in prompt for word in ["geometric", "abstract", "pattern"]):
            return "geometric"
        else:
            return "general"
    
    def _apply_watercolor_effect(self, image: Image.Image) -> Image.Image:
        """Apply watercolor painting effect"""
        # Create a softer, more artistic version
        
        # Apply gaussian blur for softness
        blurred = image.filter(ImageFilter.GaussianBlur(radius=2))
        
        # Enhance colors slightly
        enhancer = ImageEnhance.Color(blurred)
        enhanced = enhancer.enhance(1.3)
        
        # Add some texture by blending with a slightly different version
        texture = enhanced.filter(ImageFilter.EDGE_ENHANCE_MORE)
        
        # Blend the images
        result = Image.blend(enhanced, texture, 0.1)
        
        # Apply pastel color overlay
        overlay = self._create_pastel_overlay(result.size)
        result = Image.blend(result, overlay, 0.15)
        
        # Add subtle vignette
        result = self._add_vignette(result)
        
        return result
    
    def _apply_digital_art_effect(self, image: Image.Image) -> Image.Image:
        """Apply vibrant digital art effect"""
        # Enhance contrast and saturation
        contrast_enhancer = ImageEnhance.Contrast(image)
        contrasted = contrast_enhancer.enhance(1.4)
        
        color_enhancer = ImageEnhance.Color(contrasted)
        vibrant = color_enhancer.enhance(1.6)
        
        # Apply edge enhancement
        edges = vibrant.filter(ImageFilter.EDGE_ENHANCE_MORE)
        
        # Add some glow effect
        glow = edges.filter(ImageFilter.GaussianBlur(radius=3))
        result = Image.blend(edges, glow, 0.3)
        
        # Apply color shift for digital feel
        result = self._apply_color_shift(result, hue_shift=0.1)
        
        return result
    
    def _apply_minimalist_effect(self, image: Image.Image) -> Image.Image:
        """Apply minimalist line art effect"""
        # Convert to grayscale first
        gray = image.convert('L')
        
        # Apply edge detection
        edges = gray.filter(ImageFilter.FIND_EDGES)
        
        # Invert to get black lines on white
        inverted = ImageOps.invert(edges)
        
        # Convert back to RGB
        result = inverted.convert('RGB')
        
        # Apply slight color tint
        tint_color = random.choice(self.pastel_colors)
        tinted = self._apply_color_tint(result, tint_color, 0.2)
        
        return tinted
    
    def _apply_vintage_effect(self, image: Image.Image) -> Image.Image:
        """Apply vintage poster effect"""
        # Reduce colors for poster-like effect
        posterized = image.quantize(colors=8).convert('RGB')
        
        # Apply sepia tone
        sepia = self._apply_sepia(posterized)
        
        # Add some noise/grain
        noisy = self._add_noise(sepia, intensity=0.1)
        
        # Slightly desaturate
        desaturator = ImageEnhance.Color(noisy)
        result = desaturator.enhance(0.8)
        
        # Add vignette
        result = self._add_vignette(result, intensity=0.3)
        
        return result
    
    def _apply_geometric_effect(self, image: Image.Image) -> Image.Image:
        """Apply geometric/abstract effect"""
        # Create a geometric overlay
        geometric_overlay = self._create_geometric_overlay(image.size)
        
        # Blend with original
        blended = Image.blend(image, geometric_overlay, 0.4)
        
        # Apply posterization
        posterized = blended.quantize(colors=12).convert('RGB')
        
        # Enhance contrast
        contrast_enhancer = ImageEnhance.Contrast(posterized)
        result = contrast_enhancer.enhance(1.3)
        
        return result
    
    def _apply_general_artistic_effect(self, image: Image.Image) -> Image.Image:
        """Apply a general artistic transformation"""
        # Combine multiple subtle effects
        
        # Slight blur for softness
        soft = image.filter(ImageFilter.GaussianBlur(radius=1))
        
        # Enhance colors
        color_enhancer = ImageEnhance.Color(soft)
        colorful = color_enhancer.enhance(1.2)
        
        # Add slight edge enhancement
        edges = colorful.filter(ImageFilter.EDGE_ENHANCE)
        
        # Apply pastel overlay
        overlay = self._create_pastel_overlay(edges.size)
        result = Image.blend(edges, overlay, 0.1)
        
        return result
    
    def _create_pastel_overlay(self, size: Tuple[int, int]) -> Image.Image:
        """Create a pastel color overlay"""
        overlay = Image.new('RGB', size, random.choice(self.pastel_colors))
        return overlay
    
    def _add_vignette(self, image: Image.Image, intensity: float = 0.2) -> Image.Image:
        """Add a subtle vignette effect"""
        width, height = image.size
        
        # Create vignette mask
        vignette = Image.new('L', (width, height), 255)
        draw = ImageDraw.Draw(vignette)
        
        # Create gradient from center
        center_x, center_y = width // 2, height // 2
        max_radius = min(width, height) // 2
        
        for i in range(max_radius):
            alpha = int(255 * (1 - (i / max_radius) * intensity))
            draw.ellipse([
                center_x - i, center_y - i,
                center_x + i, center_y + i
            ], fill=alpha)
        
        # Apply vignette
        vignette_img = Image.new('RGB', image.size, (0, 0, 0))
        result = Image.composite(image, vignette_img, vignette)
        
        return result
    
    def _apply_color_shift(self, image: Image.Image, hue_shift: float) -> Image.Image:
        """Apply a hue shift to the image"""
        # Convert to HSV, shift hue, convert back
        hsv = image.convert('HSV')
        h, s, v = hsv.split()
        
        # Shift hue
        h_array = np.array(h)
        h_array = (h_array + int(hue_shift * 255)) % 255
        h = Image.fromarray(h_array.astype('uint8'))
        
        # Recombine
        result = Image.merge('HSV', (h, s, v)).convert('RGB')
        return result
    
    def _apply_color_tint(self, image: Image.Image, tint_color: Tuple[int, int, int], intensity: float) -> Image.Image:
        """Apply a color tint to the image"""
        tint = Image.new('RGB', image.size, tint_color)
        result = Image.blend(image, tint, intensity)
        return result
    
    def _apply_sepia(self, image: Image.Image) -> Image.Image:
        """Apply sepia tone effect"""
        pixels = image.load()
        width, height = image.size
        
        for x in range(width):
            for y in range(height):
                r, g, b = pixels[x, y]
                
                # Sepia formula
                tr = int(0.393 * r + 0.769 * g + 0.189 * b)
                tg = int(0.349 * r + 0.686 * g + 0.168 * b)
                tb = int(0.272 * r + 0.534 * g + 0.131 * b)
                
                # Clamp values
                tr = min(255, tr)
                tg = min(255, tg)
                tb = min(255, tb)
                
                pixels[x, y] = (tr, tg, tb)
        
        return image
    
    def _add_noise(self, image: Image.Image, intensity: float = 0.1) -> Image.Image:
        """Add subtle noise to the image"""
        noise = Image.new('RGB', image.size)
        pixels = noise.load()
        
        for x in range(image.size[0]):
            for y in range(image.size[1]):
                noise_val = random.randint(-int(intensity * 255), int(intensity * 255))
                pixels[x, y] = (noise_val, noise_val, noise_val)
        
        result = Image.blend(image, noise, 0.1)
        return result
    
    def _create_geometric_overlay(self, size: Tuple[int, int]) -> Image.Image:
        """Create a geometric pattern overlay"""
        overlay = Image.new('RGB', size, (255, 255, 255))
        draw = ImageDraw.Draw(overlay)
        
        # Draw random geometric shapes
        colors = self.pastel_colors
        
        for _ in range(random.randint(5, 15)):
            color = random.choice(colors)
            shape_type = random.choice(['rectangle', 'ellipse', 'polygon'])
            
            if shape_type == 'rectangle':
                x1 = random.randint(0, size[0])
                y1 = random.randint(0, size[1])
                x2 = random.randint(x1, size[0])
                y2 = random.randint(y1, size[1])
                draw.rectangle([x1, y1, x2, y2], fill=color)
            
            elif shape_type == 'ellipse':
                x1 = random.randint(0, size[0])
                y1 = random.randint(0, size[1])
                x2 = random.randint(x1, size[0])
                y2 = random.randint(y1, size[1])
                draw.ellipse([x1, y1, x2, y2], fill=color)
        
        return overlay