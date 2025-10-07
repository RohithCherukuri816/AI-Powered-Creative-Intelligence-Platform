import logging
import os
import random
from typing import Optional

import numpy as np
import torch
from PIL import Image, ImageFilter, ImageEnhance, ImageOps, ImageDraw

logger = logging.getLogger(__name__)

class ImageProcessor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._ai_loaded = False
        self._pipe = None
        self._controlnet = None
        self._free_pipe = None

    def apply_style_transformation(self, image: Image.Image, prompt: str, generation_id: str, 
                                 recognized_label: Optional[str] = None) -> Image.Image:
        """Main method for image generation - now uses recognition results"""
        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Use recognition label to enhance the prompt
            enhanced_prompt = self._enhance_prompt_with_label(prompt, recognized_label)
            
            # Generate based on recognized label rather than exact doodle
            result = self._generate_from_label(enhanced_prompt, generation_id, image)
            
            if result is not None:
                return result
            return self._fallback(image)
        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            return self._fallback(image)

    def _enhance_prompt_with_label(self, prompt: str, label: Optional[str]) -> str:
        """Enhance user prompt with recognized label"""
        if label:
            # Add the recognized object to prompt
            base_prompt = f"{prompt} {label}"
        else:
            base_prompt = prompt
        
        # Add quality terms
        quality_terms = "masterpiece, best quality, highly detailed, sharp focus, professional photography"
        style_terms = "photorealistic, realistic, natural lighting, 35mm photograph"
        
        return f"{base_prompt}, {quality_terms}, {style_terms}"

    def _generate_from_label(self, prompt: str, generation_id: str, 
                           original_image: Optional[Image.Image] = None) -> Optional[Image.Image]:
        """Generate image based on label with optional doodle guidance"""
        if not torch.cuda.is_available():
            return None
        
        if not self._load_ai():
            return None
        
        try:
            # Use random seed for variety
            seed = random.randint(0, 2**32 - 1)
            gen = torch.Generator(device="cuda").manual_seed(seed)
            
            negative_prompt = (
                "cartoon, drawing, sketch, painting, anime, 2D, flat, vector, "
                "low quality, worst quality, blurry, pixelated, grainy, "
                "deformed, distorted, disfigured, bad anatomy, "
                "text, watermark, signature, logo, frame, border"
            )

            # Parameters optimized for creative generation
            steps = int(os.getenv("NUM_STEPS", "25"))
            guidance = float(os.getenv("GUIDANCE", "7.5"))
            
            print(f"Generating from label: {prompt}")

            # Use free-form generation without strict ControlNet constraints
            if not hasattr(self, '_free_pipe') or self._free_pipe is None:
                from diffusers import StableDiffusionPipeline
                self._free_pipe = StableDiffusionPipeline.from_pretrained(
                    os.getenv("STABLE_DIFFUSION_MODEL", "runwayml/stable-diffusion-v1-5"),
                    torch_dtype=torch.float16,
                    use_safetensors=True,
                    safety_checker=None,
                    requires_safety_checker=False
                )
                self._free_pipe.to(self.device)
                
                # Optimizations
                if hasattr(self._free_pipe, "enable_attention_slicing"):
                    self._free_pipe.enable_attention_slicing()

            # Generate image
            result = self._free_pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=gen,
                height=512,
                width=512,
            ).images[0]
            
            return result
            
        except Exception as e:
            logger.error(f"Free-form generation error: {e}")
            return None

    def _load_ai(self) -> bool:
        """Load AI models (simplified - only need SD now)"""
        if self._ai_loaded:
            return True
        
        try:
            # We only need Stable Diffusion now, not ControlNet
            from diffusers import StableDiffusionPipeline
            
            self._free_pipe = StableDiffusionPipeline.from_pretrained(
                os.getenv("STABLE_DIFFUSION_MODEL", "runwayml/stable-diffusion-v1-5"),
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                use_safetensors=True,
                safety_checker=None,
                requires_safety_checker=False
            )
            self._free_pipe.to(self.device)
            
            # VRAM optimizations
            if hasattr(self._free_pipe, "enable_attention_slicing"):
                self._free_pipe.enable_attention_slicing()
            if hasattr(self._free_pipe, "enable_xformers_memory_efficient_attention"):
                try:
                    self._free_pipe.enable_xformers_memory_efficient_attention()
                except Exception:
                    pass
            
            self._ai_loaded = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to load AI models: {e}")
            return False

    def _fallback(self, image: Image.Image) -> Image.Image:
        """Enhanced fallback processing"""
        # Apply artistic filters to make doodle more interesting
        colorful = ImageEnhance.Color(image).enhance(1.6)
        contrast = ImageEnhance.Contrast(colorful).enhance(1.3)
        bright = ImageEnhance.Brightness(contrast).enhance(1.1)
        artistic = bright.filter(ImageFilter.SMOOTH_MORE)
        
        return artistic