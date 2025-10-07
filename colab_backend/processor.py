import logging
import os
import random
from typing import Optional

import numpy as np
import torch
from PIL import Image, ImageFilter, ImageEnhance, ImageOps, ImageDraw
import cv2

logger = logging.getLogger(__name__)

class ImageProcessor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._ai_loaded = False
        self._pipe = None
        self._controlnet = None
        
        print(f"🎯 ImageProcessor initialized on: {self.device}")

    def apply_style_transformation(self, image: Image.Image, prompt: str, generation_id: str, 
                                 recognized_label: Optional[str] = None) -> Image.Image:
        """
        URGENT FIX: Handle 'generate original image' case properly
        """
        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            print(f"🎨 USER PROMPT: '{prompt}'")
            print(f"🔍 ORIGINAL LABEL: '{recognized_label}'")
            
            # EMERGENCY FIX: Special handling for "generate original image" prompts
            final_label = self._handle_original_image_request(image, prompt, recognized_label)
            
            # Resize for processing
            processed_image = self._resize_for_sd(image, 512)
            
            # Generate with emergency fix
            result = self._generate_with_emergency_fix(processed_image, prompt, final_label, generation_id)
            
            if result is not None:
                print(f"✅ SUCCESS: Generated image based on doodle!")
                return result
            
            print("⚠️ AI generation failed, using fallback")
            return self._fallback(image)
            
        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            print(f"❌ ERROR: {e}")
            return self._fallback(image)

    def _handle_original_image_request(self, image: Image.Image, prompt: str, original_label: Optional[str]) -> str:
        """
        URGENT FIX: Handle cases where user wants the original doodle image
        """
        prompt_lower = prompt.lower()
        
        # Check if user wants the original doodle
        original_keywords = ['original image', 'original doodle', 'drawn doodle', 'my drawing', 'same image']
        
        if any(keyword in prompt_lower for keyword in original_keywords):
            print("🎯 DETECTED: User wants original doodle enhanced")
            # Use simple shape analysis to avoid house default
            return self._simple_shape_analysis(image)
        
        # For other prompts, use enhanced recognition
        return self._enhanced_recognition(image, original_label)

    def _simple_shape_analysis(self, image: Image.Image) -> str:
        """
        Simple reliable shape analysis to avoid house default
        """
        try:
            width, height = image.size
            aspect_ratio = width / height
            
            print(f"📐 SIMPLE SHAPE ANALYSIS: {width}x{height}, ratio: {aspect_ratio:.2f}")
            
            # Very simple logic - just avoid house default
            if aspect_ratio > 1.5:
                return "vehicle"
            elif aspect_ratio < 0.7:
                return "tree" 
            elif 0.9 < aspect_ratio < 1.1:
                return "object"  # Avoid house!
            else:
                return "object"
                
        except Exception as e:
            print(f"❌ Simple shape analysis failed: {e}")
            return "object"

    def _enhanced_recognition(self, image: Image.Image, original_label: Optional[str]) -> str:
        """
        Enhanced recognition but with house detection prevention
        """
        try:
            width, height = image.size
            aspect_ratio = width / height
            
            print(f"📊 ENHANCED ANALYSIS: {width}x{height}, ratio: {aspect_ratio:.2f}")
            
            # PREVENT HOUSE DEFAULT - be very conservative
            if aspect_ratio > 1.8:
                return "van"
            elif aspect_ratio > 1.5:
                return "car"
            elif aspect_ratio > 2.2:
                return "bus"
            elif aspect_ratio < 0.6:
                return "tree"
            elif 0.95 < aspect_ratio < 1.05:
                # Only return house if it's very square AND we're confident
                return "object"  # Avoid house!
            else:
                return original_label or "object"
                
        except Exception as e:
            print(f"❌ Enhanced recognition failed: {e}")
            return "object"

    def _generate_with_emergency_fix(self, image: Image.Image, user_prompt: str, label: str, generation_id: str) -> Optional[Image.Image]:
        """
        Generation with emergency fixes to prevent house images
        """
        if not torch.cuda.is_available():
            print("❌ CUDA not available for AI generation")
            return None
        
        if not self._load_ai():
            print("❌ Failed to load AI models")
            return None
        
        try:
            # Prepare control image
            control_image = self._prepare_control_image(image)
            
            # EMERGENCY FIX: Special prompt handling for "original image" requests
            prompt_lower = user_prompt.lower()
            if any(keyword in prompt_lower for keyword in ['original image', 'original doodle', 'drawn doodle']):
                enhanced_prompt = self._create_original_image_prompt(user_prompt, label)
            else:
                enhanced_prompt = self._create_optimized_prompt(user_prompt, label)
            
            # Generation parameters - optimized for doodle following
            seed = random.randint(0, 2**32 - 1)
            gen = torch.Generator(device="cuda").manual_seed(seed)
            
            negative_prompt = (
                "house, building, architecture, "  # EXPLICITLY BLOCK HOUSE!
                "cartoon, drawing, sketch, painting, anime, 2D, flat, vector, "
                "low quality, worst quality, blurry, pixelated, grainy, "
                "deformed, distorted, disfigured, bad anatomy, "
                "text, watermark, signature, logo, ugly, poorly drawn"
            )

            # Parameters that favor following the doodle
            steps = 25
            guidance = 7.0
            conditioning_scale = 0.8  # Higher to follow doodle more closely

            print(f"🎨 GENERATING: {enhanced_prompt}")
            print(f"⚙️ PARAMS: steps={steps}, guidance={guidance}, conditioning={conditioning_scale}")
            print(f"🚫 NEGATIVE: Explicitly blocking house generation!")

            # Generate image
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
            logger.error(f"Generation error: {e}")
            print(f"❌ GENERATION ERROR: {e}")
            return None

    def _create_original_image_prompt(self, user_prompt: str, label: str) -> str:
        """
        Special prompt for 'generate original image' requests
        """
        # Keep it simple - focus on enhancing the doodle
        base_prompt = "realistic version of the drawing, detailed, sharp focus"
        
        # Add the recognized object type if we have one
        if label != "object":
            base_prompt = f"realistic {label}, {base_prompt}"
        
        quality_terms = "photorealistic, high quality, natural lighting, professional"
        
        return f"{base_prompt}, {quality_terms}"

    def _create_optimized_prompt(self, user_prompt: str, label: str) -> str:
        """
        Create optimized prompt for normal requests
        """
        user_prompt_clean = user_prompt.strip()
        
        # Vehicle-specific enhancements
        if label.lower() in ['van', 'car', 'bus', 'truck', 'vehicle']:
            vehicle_terms = ['van', 'car', 'vehicle', 'automobile', 'truck', 'bus', 'suv']
            has_vehicle_term = any(term in user_prompt_clean.lower() for term in vehicle_terms)
            
            if has_vehicle_term:
                base_prompt = user_prompt_clean
            else:
                base_prompt = f"{user_prompt_clean} {label}"
            
            quality_terms = "photorealistic, professional vehicle photography, detailed"
            
        else:
            base_prompt = f"{user_prompt_clean} {label}" if label != "object" else user_prompt_clean
            quality_terms = "photorealistic, high quality, detailed, realistic"
        
        style_terms = "natural lighting, sharp focus, professional"
        
        return f"{base_prompt}, {quality_terms}, {style_terms}"

    def _prepare_control_image(self, image: Image.Image) -> Image.Image:
        """
        Prepare doodle image for ControlNet processing
        """
        try:
            # Convert to grayscale
            if image.mode != 'L':
                gray_image = image.convert('L')
            else:
                gray_image = image
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(gray_image)
            enhanced = enhancer.enhance(2.5)
            
            return enhanced.convert('RGB')
            
        except Exception as e:
            print(f"❌ Control image preparation failed: {e}")
            return image.convert('RGB')

    def _load_ai(self) -> bool:
        """
        Load AI models
        """
        if self._ai_loaded:
            return True
        
        try:
            from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

            print("🔄 LOADING AI MODELS...")
            
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            
            # Load ControlNet
            self._controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/control_v11p_sd15_scribble",
                torch_dtype=dtype,
            )
            
            # Load Stable Diffusion
            self._pipe = StableDiffusionControlNetPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                controlnet=self._controlnet,
                torch_dtype=dtype,
                safety_checker=None,
                requires_safety_checker=False,
            )
            
            self._pipe.to(self.device)
            
            # Optimizations
            self._pipe.enable_attention_slicing()
            
            self._ai_loaded = True
            print("✅ AI MODELS LOADED!")
            return True
            
        except Exception as e:
            print(f"❌ FAILED TO LOAD AI: {e}")
            return False

    def _resize_for_sd(self, image: Image.Image, target: int) -> Image.Image:
        """
        Resize image for Stable Diffusion
        """
        w, h = image.size
        
        if w > h:
            new_w = target
            new_h = int(h * target / w)
        else:
            new_h = target
            new_w = int(w * target / h)
        
        new_w = (new_w // 8) * 8
        new_h = (new_h // 8) * 8
        
        return image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    def _fallback(self, image: Image.Image) -> Image.Image:
        """
        Enhanced fallback
        """
        print("🔄 Using enhanced fallback")
        colorful = ImageEnhance.Color(image).enhance(1.8)
        contrast = ImageEnhance.Contrast(colorful).enhance(1.4)
        return contrast