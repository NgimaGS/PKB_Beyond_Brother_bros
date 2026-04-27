# 🎨 Image Intelligence Layer

The Nexus Engine integrates local image synthesis and visual analysis to provide a full multi-modal intelligence experience without external API dependencies.

---

## 🏗️ Technical Architecture

The image layer is managed by the `LocalImageService`, which isolates heavy ML dependencies (Torch, Diffusers) from the main UI thread.

### 1. Key Components
- **Inference Engine**: Stable Diffusion via the `diffusers` library.
- **Vision Model**: `llava:latest` running via Ollama.
- **VRAM Optimizations**: Float16 precision and attention slicing are enabled by default on CUDA devices to maximize performance on consumer hardware.

---

## 🎨 Supported Diffusion Models

| Model | ID | Rationale |
| :--- | :--- | :--- |
| **SD v1.5** | `runwayml/stable-diffusion-v1-5` | The global standard for reliable, low-VRAM generation. |
| **SDXL Turbo** | `stabilityai/sdxl-turbo` | Optimized for speed; generates high-quality images in 1-4 steps. |
| **OpenJourney** | `prompthero/openjourney` | Fine-tuned for Midjourney-style artistic aesthetics. |

---

## 👁️ Visual Intelligence (Vision Layer)

The engine uses **Vision-Augmented RAG** to understand the contents of images.
- **Automatic Captioning**: When images are uploaded or generated, they can be processed by Llava to create a textual description.
- **Semantic Indexing**: These descriptions are fed into the `KnowledgeBase`, allowing you to find images by searching for their content (e.g., searching for "cyberpunk city" will find images containing those elements).

---

## ⚠️ Troubleshooting & Hardware Tips

### 1. CUDA Out of Memory (OOM)
If you encounter OOM errors:
- Reduce the **Generation Steps** in the Image Studio settings.
- Ensure no other memory-heavy processes are utilizing the GPU.
- The system will attempt a fallback to **CPU**, but generation will be significantly slower.

### 2. Model Downloads
On the first run, the engine will download model weights from Hugging Face. Ensure you have at least **10GB of free disk space** for the Stable Diffusion cache.

---
*Core Service Implementation: [core/image_service.py](../core/image_service.py)*
