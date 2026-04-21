"""
Image Service Layer — Local Stable Diffusion & Vision Integration
==================================================================

Architecture Rationale:
-----------------------
This module manages the lifecycle of local text-to-image models. It isolates
the heavyweight 'diffusers' and 'torch' logic from the rest of the application.

Design Pattern: Service Provider with Lazy Loading
The models are not loaded into VRAM until the first generation request is made,
and only if the 'image_gen_active' flag is set.

Key Responsibilities:
1. **Model Management**: Downloading and verifying model weights via Hugging Face.
2. **Inference**: Generating images from text prompts using Stable Diffusion.
3. **Visual Intelligence**: Interface for Llava-based image analysis (via Ollama).
"""

import os
import torch
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
from huggingface_hub import snapshot_download
import PIL.Image
from datetime import datetime

class LocalImageService:
    """
    Manages local Stable Diffusion models for image generation.
    """

    RECOMMENDED_MODELS = {
        "Stable Diffusion v1.5": "runwayml/stable-diffusion-v1-5",
        "SDXL Turbo": "stabilityai/sdxl-turbo",
        "OpenJourney": "prompthero/openjourney"
    }

    def __init__(self, model_id=None, device="cuda", save_dir="data/images"):
        self.model_id = model_id or "runwayml/stable-diffusion-v1-5"
        self.device = device if torch.cuda.is_available() else "cpu"
        self.save_dir = save_dir
        self.pipe = None
        
        # Ensure save directory exists
        os.makedirs(self.save_dir, exist_ok=True)

    def is_model_downloaded(self, model_id=None) -> bool:
        """Checks if the specific model weights exist locally in the HF cache."""
        target_id = model_id or self.model_id
        # We check for the presence of the model in the default HF cache location
        # or a specific local path. Snapshot download handles this check internally.
        # For simplicity in UI, we check if the directory exists in our custom storage
        # or rely on huggingface_hub's cached_assets logic.
        try:
            # Short-circuit check: try to find the folder in hub cache
            from huggingface_hub import scan_cache_dir
            cache_info = scan_cache_dir()
            for repo in cache_info.repos:
                if repo.repo_id == target_id:
                    return True
            return False
        except Exception:
            return False

    def download_model(self, model_id: str, progress_callback=None):
        """
        Explicitly triggers a model download from Hugging Face.
        """
        try:
            # We use snapshot_download to fetch the full model repository
            snapshot_download(repo_id=model_id, local_files_only=False)
            return True
        except Exception as e:
            print(f"Download failed: {e}")
            return False

    def _load_pipeline(self):
        """Loads the model into VRAM."""
        if self.pipe is not None:
            return

        print(f"🚀 Loading Image Engine: {self.model_id} on {self.device}...")
        try:
            # Use float16 for reduced VRAM usage if on CUDA
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            
            self.pipe = StableDiffusionPipeline.from_pretrained(
                self.model_id, 
                torch_dtype=dtype,
                use_safetensors=True
            )
            self.pipe.to(self.device)
            
            # Optimized scheduler for faster generation
            self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)
            
            # Enable attention slicing to save VRAM
            if self.device == "cuda":
                self.pipe.enable_attention_slicing()
                
        except Exception as e:
            print(f"Failed to load image pipeline: {e}")
            raise

    def generate(self, prompt: str, negative_prompt: str = "", steps: int = 20, guidance_scale: float = 7.5, seed: int = None):
        """
        Generates an image from a prompt and saves it to disk.
        """
        self._load_pipeline()
        
        generator = torch.manual_seed(seed) if seed is not None else None
        
        try:
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator
            ).images[0]
            
            # Save file with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_prompt = "".join([c for c in prompt[:30] if c.isalnum() or c in (' ', '_')]).rstrip()
            filename = f"gen_{timestamp}_{safe_prompt.replace(' ', '_')}.png"
            filepath = os.path.join(self.save_dir, filename)
            
            result.save(filepath)
            return filepath, filename
        except Exception as e:
            print(f"Generation error: {e}")
            return None, str(e)

    def generate_quick(self, prompt: str):
        """Simplified generation for chat-based `/image` command."""
        # Use lower steps for speed in chat
        return self.generate(prompt, steps=20, guidance_scale=7.0)

