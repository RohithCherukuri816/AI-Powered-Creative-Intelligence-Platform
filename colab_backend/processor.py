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
        self._preprocessor = None

    def apply_style_transformation(self, image: Image.Image, prompt: str, generation_id: str, 
                                 recognized_label: Optional[str] = None) -> Image.Image:
        """Generate realistic version of the ACTUAL doodle - not a new image"""
        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize for AI processing
            image = self._resize_for_sd(image, 512)
            
            print(f"🎨 Processing ACTUAL doodle - Label: {recognized_label}")
            print(f"📝 User prompt: {prompt}")
            
            # Generate realistic version of THIS doodle
            result = self._generate_from_actual_doodle(image, prompt, recognized_label, generation_id)
            
            if result is not None:
                print("✅ Successfully generated realistic version of doodle!")
                return result
            return self._fallback(image)
        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            return self._fallback(image)

    def _generate_from_actual_doodle(self, image: Image.Image, prompt: str, label: Optional[str], generation_id: str) -> Optional[Image.Image]:
        """Generate realistic version of the actual doodle using ControlNet"""
        if not torch.cuda.is_available():
            return None
        
        if not self._load_ai():
            return None
        
        try:
            # Use the ACTUAL doodle as control image
            control_image = self._preprocess_doodle_for_controlnet(image)
            
            # Enhance prompt with the recognized label
            enhanced_prompt = self._enhance_prompt_for_doodle(prompt, label)
            
            # Use random seed for variety
            seed = random.randint(0, 2**32 - 1)
            gen = torch.Generator(device="cuda").manual_seed(seed)
            
            negative_prompt = (
                "low quality, worst quality, blurry, pixelated, grainy, "
                "deformed, distorted, disfigured, bad anatomy, "
                "text, watermark, signature, logo"
            )

            # Optimized parameters for doodle-to-realistic transformation
            steps = 30
            guidance = 7.5
            conditioning_scale = 0.8  # Balance between doodle and realism

            print(f"🔄 Transforming doodle to realistic: {enhanced_prompt}")
            print(f"⚙️ Parameters: steps={steps}, guidance={guidance}, conditioning={conditioning_scale}")

            # Generate using ControlNet to follow the doodle structure
            result = self._pipe(
                prompt=enhanced_prompt,
                negative_prompt=negative_prompt,
                image=control_image,
                num_inference_steps=steps,
                guidance_scale=guidance,
                controlnet_conditioning_scale=conditioning_scale,
                generator=gen,
                height=512,
                width=512,
            ).images[0]
            
            return result
            
        except Exception as e:
            logger.error(f"Doodle transformation error: {e}")
            return None

    def _preprocess_doodle_for_controlnet(self, image: Image.Image) -> Image.Image:
        """Preprocess doodle for ControlNet - enhance edges and structure"""
        # Convert to grayscale for better edge detection
        if image.mode != 'L':
            gray_image = image.convert('L')
        else:
            gray_image = image
        
        # Enhance contrast to make doodle more clear
        enhancer = ImageEnhance.Contrast(gray_image)
        enhanced = enhancer.enhance(2.0)
        
        # Convert back to RGB for ControlNet
        rgb_image = enhanced.convert('RGB')
        
        return rgb_image

    def _enhance_prompt_for_doodle(self, prompt: str, label: Optional[str]) -> str:
        """Enhance prompt specifically for doodle transformation"""
        base_prompt = prompt
        
        # Add quality terms for realistic transformation
        quality_terms = "high quality, detailed, sharp focus, realistic textures"
        style_terms = "photorealistic, natural lighting, professional photography"
        
        enhanced = f"{base_prompt}, {quality_terms}, {style_terms}"
        print(f"📝 Enhanced prompt: {enhanced}")
        return enhanced

    def _load_ai(self) -> bool:
        """Load AI models with ControlNet for doodle transformation"""
        if self._ai_loaded:
            return True
        
        try:
            from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
            from controlnet_aux import ScribbleDetector

            dtype = torch.float16 if self.device == "cuda" else torch.float32
            
            # Load ControlNet model for scribble/doodle transformation
            print("🔄 Loading ControlNet for doodle transformation...")
            self._controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/control_v11p_sd15_scribble",
                torch_dtype=dtype,
                use_safetensors=True,
            )
            
            # Load Stable Diffusion with ControlNet
            self._pipe = StableDiffusionControlNetPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                controlnet=self._controlnet,
                torch_dtype=dtype,
                safety_checker=None,
                requires_safety_checker=False,
                use_safetensors=True,
            )
            self._pipe.to(self.device)
            
            # Load preprocessor for doodles
            self._preprocessor = ScribbleDetector()
            
            # VRAM optimizations
            if hasattr(self._pipe, "enable_attention_slicing"):
                self._pipe.enable_attention_slicing()
            if hasattr(self._pipe, "enable_xformers_memory_efficient_attention"):
                try:
                    self._pipe.enable_xformers_memory_efficient_attention()
                except Exception:
                    pass
            
            self._ai_loaded = True
            print("✅ AI Models loaded for doodle transformation")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load AI models: {e}")
            return False

    def _resize_for_sd(self, image: Image.Image, target: int) -> Image.Image:
        """Resize image for Stable Diffusion"""
        w, h = image.size
        if w > h:
            nw, nh = target, int(h * target / w)
        else:
            nh, nw = target, int(w * target / h)
        nw = (nw // 8) * 8
        nh = (nh // 8) * 8
        return image.resize((nw, nh), Image.Resampling.LANCZOS)

    def _fallback(self, image: Image.Image) -> Image.Image:
        """Enhanced fallback processing"""
        print("⚠️ Using fallback image processing")
        # Apply artistic filters to make doodle more interesting
        colorful = ImageEnhance.Color(image).enhance(1.6)
        contrast = ImageEnhance.Contrast(colorful).enhance(1.3)
        bright = ImageEnhance.Brightness(contrast).enhance(1.1)
        artistic = bright.filter(ImageFilter.SMOOTH_MORE)
        
        return artistic