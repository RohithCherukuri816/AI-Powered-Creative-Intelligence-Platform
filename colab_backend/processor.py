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
        self._preprocessor = None
        
        print(f"🎯 ImageProcessor initialized on: {self.device}")

    def apply_style_transformation(self, image: Image.Image, prompt: str, generation_id: str, 
                                 recognized_label: Optional[str] = None) -> Image.Image:
        """
        Generate realistic version of the ACTUAL doodle with enhanced recognition
        """
        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Enhanced recognition with shape analysis
            final_label = self._enhanced_recognition(image, recognized_label)
            
            print(f"🔍 ENHANCED RECOGNITION: '{final_label}'")
            print(f"📝 USER PROMPT: '{prompt}'")
            
            # Resize for processing
            processed_image = self._resize_for_sd(image, 512)
            
            # Generate realistic version
            result = self._generate_realistic_version(processed_image, prompt, final_label, generation_id)
            
            if result is not None:
                print(f"✅ SUCCESS: Generated realistic {final_label}!")
                return result
            
            print("⚠️ AI generation failed, using fallback")
            return self._fallback(image)
            
        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            print(f"❌ ERROR: {e}")
            return self._fallback(image)

    def _enhanced_recognition(self, image: Image.Image, original_label: Optional[str]) -> str:
        """
        Enhanced recognition with computer vision shape analysis
        """
        try:
            # Convert to numpy for CV analysis
            if image.mode != 'L':
                gray = image.convert('L')
            else:
                gray = image
            
            # Create a copy for analysis
            analysis_img = gray.resize((200, 200), Image.Resampling.LANCZOS)
            np_img = np.array(analysis_img)
            
            # Invert if needed (white doodle on black background)
            if np.mean(np_img) > 127:
                np_img = 255 - np_img
            
            # Apply threshold
            _, binary = cv2.threshold(np_img, 50, 255, cv2.THRESH_BINARY)
            
            # Find contours
            contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                print("📊 SHAPE ANALYSIS: No contours found")
                return original_label or "vehicle"
            
            # Analyze largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            aspect_ratio = w / h if h > 0 else 1
            area = cv2.contourArea(largest_contour)
            bbox_area = w * h
            fullness = area / bbox_area if bbox_area > 0 else 0
            
            print(f"📊 SHAPE ANALYSIS: aspect_ratio={aspect_ratio:.2f}, fullness={fullness:.2f}")
            
            # Vehicle detection logic
            if aspect_ratio > 1.8 and w > 80:
                print("🚗 VEHICLE DETECTED: Wide rectangular shape")
                return "van"
            elif aspect_ratio > 1.5 and w > 60:
                print("🚙 VEHICLE DETECTED: Rectangular shape") 
                return "car"
            elif aspect_ratio > 2.0:
                print("🚂 VEHICLE DETECTED: Very wide shape")
                return "bus"
            elif 0.8 < aspect_ratio < 1.2 and fullness > 0.6:
                print("🏠 HOUSE DETECTED: Square-ish filled shape")
                return "house"
            elif aspect_ratio < 0.7:
                print("🌳 TREE DETECTED: Tall shape")
                return "tree"
            else:
                print("❓ UNKNOWN SHAPE: Using original label")
                return original_label or "object"
                
        except Exception as e:
            print(f"❌ Shape analysis failed: {e}")
            return original_label or "vehicle"

    def _generate_realistic_version(self, image: Image.Image, user_prompt: str, label: str, generation_id: str) -> Optional[Image.Image]:
        """
        Generate realistic version using ControlNet with enhanced prompts
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
            
            # Create optimized prompt
            enhanced_prompt = self._create_optimized_prompt(user_prompt, label)
            
            # Generation parameters
            seed = random.randint(0, 2**32 - 1)
            gen = torch.Generator(device="cuda").manual_seed(seed)
            
            negative_prompt = (
                "cartoon, drawing, sketch, painting, anime, 2D, flat, vector, "
                "low quality, worst quality, blurry, pixelated, grainy, "
                "deformed, distorted, disfigured, bad anatomy, "
                "text, watermark, signature, logo, ugly, poorly drawn"
            )

            # Optimized parameters for vehicle transformation
            steps = 30
            guidance = 7.5
            conditioning_scale = 0.75  # Good balance for following doodle

            print(f"🎨 GENERATING: {enhanced_prompt}")
            print(f"⚙️ PARAMS: steps={steps}, guidance={guidance}, conditioning={conditioning_scale}")

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

    def _create_optimized_prompt(self, user_prompt: str, label: str) -> str:
        """
        Create optimized prompt that combines user intent with recognized object
        """
        # Clean user prompt
        user_prompt_clean = user_prompt.strip()
        
        # Vehicle-specific enhancements
        if label.lower() in ['van', 'car', 'bus', 'truck', 'vehicle']:
            # Check if user already mentioned a vehicle
            vehicle_terms = ['van', 'car', 'vehicle', 'automobile', 'truck', 'bus', 'suv']
            has_vehicle_term = any(term in user_prompt_clean.lower() for term in vehicle_terms)
            
            if has_vehicle_term:
                # User mentioned a vehicle, use their prompt directly
                base_prompt = user_prompt_clean
            else:
                # User didn't mention vehicle, add the recognized type
                base_prompt = f"{user_prompt_clean} {label}"
            
            # Add vehicle-specific quality terms
            quality_terms = "photorealistic, professional vehicle photography, detailed, sharp focus"
            style_terms = "realistic, natural lighting, high resolution, clean design"
            
        else:
            # For non-vehicles
            base_prompt = f"{user_prompt_clean} {label}" if label != "object" else user_prompt_clean
            quality_terms = "photorealistic, high quality, detailed, sharp focus"
            style_terms = "realistic, natural lighting, professional photography"
        
        final_prompt = f"{base_prompt}, {quality_terms}, {style_terms}"
        return final_prompt

    def _prepare_control_image(self, image: Image.Image) -> Image.Image:
        """
        Prepare doodle image for ControlNet processing
        """
        try:
            # Convert to grayscale for better edge detection
            if image.mode != 'L':
                gray_image = image.convert('L')
            else:
                gray_image = image
            
            # Enhance contrast to make doodle more visible
            enhancer = ImageEnhance.Contrast(gray_image)
            enhanced = enhancer.enhance(2.5)  # Higher contrast for better control
            
            # Convert back to RGB for ControlNet
            rgb_image = enhanced.convert('RGB')
            
            return rgb_image
            
        except Exception as e:
            print(f"❌ Control image preparation failed: {e}")
            return image.convert('RGB')

    def _load_ai(self) -> bool:
        """
        Load AI models with proper error handling
        """
        if self._ai_loaded:
            return True
        
        try:
            from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
            from controlnet_aux import ScribbleDetector

            print("🔄 LOADING AI MODELS...")
            
            # Use appropriate dtype
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            
            # Load ControlNet for scribble/doodle transformation
            print("📥 Loading ControlNet...")
            self._controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/control_v11p_sd15_scribble",
                torch_dtype=dtype,
                use_safetensors=True,
            )
            
            # Load Stable Diffusion pipeline with ControlNet
            print("📥 Loading Stable Diffusion...")
            self._pipe = StableDiffusionControlNetPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                controlnet=self._controlnet,
                torch_dtype=dtype,
                safety_checker=None,
                requires_safety_checker=False,
                use_safetensors=True,
            )
            
            # Move to device
            self._pipe.to(self.device)
            print(f"✅ Models moved to: {self.device}")
            
            # Load preprocessor
            print("📥 Loading preprocessor...")
            self._preprocessor = ScribbleDetector()
            
            # Apply optimizations
            print("⚙️ Applying optimizations...")
            if hasattr(self._pipe, "enable_attention_slicing"):
                self._pipe.enable_attention_slicing()
                print("✅ Attention slicing enabled")
            
            if hasattr(self._pipe, "enable_xformers_memory_efficient_attention"):
                try:
                    self._pipe.enable_xformers_memory_efficient_attention()
                    print("✅ XFormers enabled")
                except Exception as e:
                    print(f"⚠️ XFormers not available: {e}")
            
            self._ai_loaded = True
            print("🎉 AI MODELS LOADED SUCCESSFULLY!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load AI models: {e}")
            print(f"❌ FAILED TO LOAD AI MODELS: {e}")
            return False

    def _resize_for_sd(self, image: Image.Image, target: int) -> Image.Image:
        """
        Resize image for Stable Diffusion while maintaining aspect ratio
        """
        w, h = image.size
        
        # Calculate new dimensions maintaining aspect ratio
        if w > h:
            new_w = target
            new_h = int(h * target / w)
        else:
            new_h = target
            new_w = int(w * target / h)
        
        # Ensure dimensions are multiples of 8
        new_w = (new_w // 8) * 8
        new_h = (new_h // 8) * 8
        
        # Resize with high-quality filter
        resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        print(f"📐 Resized: {w}x{h} -> {new_w}x{new_h}")
        return resized

    def _fallback(self, image: Image.Image) -> Image.Image:
        """
        Enhanced fallback processing when AI generation fails
        """
        print("🔄 Using enhanced fallback processing")
        
        try:
            # Create an enhanced version of the original doodle
            colorful = ImageEnhance.Color(image).enhance(1.8)
            contrast = ImageEnhance.Contrast(colorful).enhance(1.4)
            bright = ImageEnhance.Brightness(contrast).enhance(1.2)
            
            # Apply artistic filters
            smoothed = bright.filter(ImageFilter.SMOOTH_MORE)
            sharpened = smoothed.filter(ImageFilter.SHARPEN)
            
            print("✅ Fallback processing completed")
            return sharpened
            
        except Exception as e:
            print(f"❌ Fallback processing failed: {e}")
            return image

    def cleanup(self):
        """
        Clean up GPU memory
        """
        try:
            if self._pipe is not None:
                del self._pipe
            if self._controlnet is not None:
                del self._controlnet
                
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            print("🧹 GPU memory cleaned up")
            
        except Exception as e:
            print(f"⚠️ Cleanup failed: {e}")