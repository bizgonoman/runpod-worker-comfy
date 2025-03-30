# Stage 1: Base image with common dependencies
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 as base

# Prevents prompts from packages asking for user input during installation
ENV DEBIAN_FRONTEND=noninteractive
# Prefer binary wheels over source distributions for faster pip installations
ENV PIP_PREFER_BINARY=1
# Ensures output from python is printed immediately to the terminal without buffering
ENV PYTHONUNBUFFERED=1 
# Speed up some cmake builds
ENV CMAKE_BUILD_PARALLEL_LEVEL=8

# Install Python, git and other necessary tools
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    libgl1 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip



# Clean up to reduce image size
RUN apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Install comfy-cli
RUN pip install comfy-cli

# Install ComfyUI
RUN /usr/bin/yes | comfy --workspace /comfyui install --cuda-version 11.8 --nvidia --version 0.3.26

# Change working directory to ComfyUI
WORKDIR /comfyui

# Install runpod
RUN pip install runpod requests

# Support for the network volume
ADD src/extra_model_paths.yaml ./

# Go back to the root
WORKDIR /

# Add scripts
ADD src/start.sh src/restore_snapshot.sh src/rp_handler.py test_input.json ./
RUN chmod +x /start.sh /restore_snapshot.sh

# Optionally copy the snapshot file
ADD *snapshot*.json /

# Restore the snapshot to install custom nodes
RUN /restore_snapshot.sh

# Start container
CMD ["/start.sh"]

# Stage 2: Download models
FROM base as downloader

ARG HUGGINGFACE_ACCESS_TOKEN
ARG MODEL_TYPE

# Change working directory to ComfyUI
WORKDIR /comfyui

# Create necessary directories
RUN mkdir -p models/checkpoints models/vae models/controlnet models/embeddings models/grounding-dino models/ipadapter models/clip_vision models/sam2 models/vae_approx

# Download checkpoints/vae/LoRA to include in image based on model type
RUN wget -O models/checkpoints/off2onsPDXLRealistic_1a1.safetensors https://huggingface.co/bizgonoman/kmd-model/resolve/main/checkpoint/off2onsPDXLRealistic_1a1.safetensors
RUN wget -O models/vae/sdxlVAE_sdxlVAE.safetensors https://huggingface.co/bizgonoman/kmd-model/resolve/main/vae/sdxlVAE_sdxlVAE.safetensors
RUN wget -O models/vae/sdxlNaturalSkintone_fp32.safetensors https://huggingface.co/bizgonoman/kmd-model/resolve/main/vae/sdxlNaturalSkintone_fp32.safetensors
RUN wget -O models/controlnet/control-lora-openposeXL2-rank256.safetensors https://huggingface.co/bizgonoman/kmd-model/resolve/main/controlnet/control-lora-openposeXL2-rank256.safetensors
RUN wget -O models/embeddings/pony_female_neg.safetensors https://huggingface.co/bizgonoman/kmd-model/resolve/main/embeddings/pony_female_neg.safetensors
RUN wget -O models/embeddings/pony_hq_v1_neg.pt https://huggingface.co/bizgonoman/kmd-model/resolve/main/embeddings/pony_hq_v1_neg.pt
RUN wget -O models/embeddings/pony_hq_v1_pos.pt https://huggingface.co/bizgonoman/kmd-model/resolve/main/embeddings/pony_hq_v1_pos.pt
RUN wget -O models/embeddings/pony_hq_v2_neg.safetensors https://huggingface.co/bizgonoman/kmd-model/resolve/main/embeddings/pony_hq_v2_neg.safetensors
RUN wget -O models/embeddings/pony_hq_v2_pos.safetensors https://huggingface.co/bizgonoman/kmd-model/resolve/main/embeddings/pony_hq_v2_pos.safetensors
RUN wget -O models/embeddings/pony_photo_real_neg.pt https://huggingface.co/bizgonoman/kmd-model/resolve/main/embeddings/pony_photo_real_neg.pt
RUN wget -O models/embeddings/pony_photo_real_pos.pt https://huggingface.co/bizgonoman/kmd-model/resolve/main/embeddings/pony_photo_real_pos.pt
RUN wget -O models/embeddings/pony_real_piano_neg.safetensors https://huggingface.co/bizgonoman/kmd-model/resolve/main/embeddings/pony_real_piano_neg.safetensors
RUN wget -O models/embeddings/pony_xxx_rate_neg.pt https://huggingface.co/bizgonoman/kmd-model/resolve/main/embeddings/pony_xxx_rate_neg.pt
RUN wget -O models/embeddings/pony_xxx_rate_pos.pt https://huggingface.co/bizgonoman/kmd-model/resolve/main/embeddings/pony_xxx_rate_pos.pt
RUN wget -O models/embeddings/vxl_analogfilm_pos.safetensors https://huggingface.co/bizgonoman/kmd-model/resolve/main/embeddings/vxl_analogfilm_pos.safetensors
RUN wget -O models/embeddings/vxl_bad_x_neg.pt https://huggingface.co/bizgonoman/kmd-model/resolve/main/embeddings/vxl_bad_x_neg.pt
RUN wget -O models/embeddings/vxl_cinematic_pos.safetensors https://huggingface.co/bizgonoman/kmd-model/resolve/main/embeddings/vxl_cinematic_pos.safetensors
RUN wget -O models/embeddings/vxl_deepng_neg.safetensors https://huggingface.co/bizgonoman/kmd-model/resolve/main/embeddings/vxl_deepng_neg.safetensors
RUN wget -O models/embeddings/vxl_dtplus_hair_pos.safetensors https://huggingface.co/bizgonoman/kmd-model/resolve/main/embeddings/vxl_dtplus_hair_pos.safetensors
RUN wget -O models/embeddings/vxl_dtplus_overall_pos.safetensors https://huggingface.co/bizgonoman/kmd-model/resolve/main/embeddings/vxl_dtplus_overall_pos.safetensors
RUN wget -O models/embeddings/vxl_dtplus_skin_pos.safetensors https://huggingface.co/bizgonoman/kmd-model/resolve/main/embeddings/vxl_dtplus_skin_pos.safetensors
RUN wget -O models/embeddings/vxl_endless_up_neg.safetensors https://huggingface.co/bizgonoman/kmd-model/resolve/main/embeddings/vxl_endless_up_neg.safetensors
RUN wget -O models/embeddings/vxl_fix_blur_neg.safetensors https://huggingface.co/bizgonoman/kmd-model/resolve/main/embeddings/vxl_fix_blur_neg.safetensors
RUN wget -O models/embeddings/vxl_general_neg.safetensors https://huggingface.co/bizgonoman/kmd-model/resolve/main/embeddings/vxl_general_neg.safetensors
RUN wget -O models/embeddings/vxl_negative_ti_alb_neg.safetensors https://huggingface.co/bizgonoman/kmd-model/resolve/main/embeddings/vxl_negative_ti_alb_neg.safetensors
RUN wget -O models/embeddings/vxl_realism_neg.safetensors https://huggingface.co/bizgonoman/kmd-model/resolve/main/embeddings/vxl_realism_neg.safetensors
RUN wget -O models/embeddings/vxl_realism_pos.safetensors https://huggingface.co/bizgonoman/kmd-model/resolve/main/embeddings/vxl_realism_pos.safetensors
RUN wget -O models/grounding-dino/GroundingDINO_SwinB.cfg.py https://huggingface.co/bizgonoman/kmd-model/resolve/main/grounding-dino/GroundingDINO_SwinB.cfg.py
RUN wget -O models/grounding-dino/GroundingDINO_SwinT_OGC.cfg.py https://huggingface.co/bizgonoman/kmd-model/resolve/main/grounding-dino/GroundingDINO_SwinT_OGC.cfg.py
RUN wget -O models/grounding-dino/groundingdino_swinb_cogcoor.pth https://huggingface.co/bizgonoman/kmd-model/resolve/main/grounding-dino/groundingdino_swinb_cogcoor.pth
RUN wget -O models/grounding-dino/groundingdino_swint_ogc.pth https://huggingface.co/bizgonoman/kmd-model/resolve/main/grounding-dino/groundingdino_swint_ogc.pth
RUN wget -O models/ipadapter/groundingdino_swint_ogc.pth https://huggingface.co/bizgonoman/kmd-model/resolve/main/ipadapter/ip-adapter-faceid-plusv2_sdxl.bin
RUN wget -O models/ipadapter/groundingdino_swint_ogc.pth https://huggingface.co/bizgonoman/kmd-model/resolve/main/ipadapter/ip-adapter-faceid_sdxl.bin
RUN wget -O models/clip_vision/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors https://huggingface.co/bizgonoman/kmd-model/resolve/main/clip_vision/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors.safetensors
RUN wget -O models/sam2/sam2.1_hiera_base_plus.pt https://huggingface.co/bizgonoman/kmd-model/resolve/main/sam2/sam2.1_hiera_base_plus.pt
RUN wget -O models/sam2/sam2.1_hiera_large.pt https://huggingface.co/bizgonoman/kmd-model/resolve/main/sam2/sam2.1_hiera_large.pt
RUN wget -O models/sam2/sam2.1_hiera_small.pt https://huggingface.co/bizgonoman/kmd-model/resolve/main/sam2/sam2.1_hiera_small.pt
RUN wget -O models/sam2/sam2.1_hiera_tiny.pt https://huggingface.co/bizgonoman/kmd-model/resolve/main/sam2/sam2.1_hiera_tiny.pt
RUN wget -O models/sam2/sam2_hiera_base_plus.pt https://huggingface.co/bizgonoman/kmd-model/resolve/main/sam2/sam2_hiera_base_plus.pt
RUN wget -O models/sam2/sam2_hiera_large.pt https://huggingface.co/bizgonoman/kmd-model/resolve/main/sam2/sam2_hiera_large.pt
RUN wget -O models/vae_approx/taef1_decoder.pth https://huggingface.co/bizgonoman/kmd-model/resolve/main/vae_approx/taef1_decoder.pth
RUN wget -O models/vae_approx/taef1_encoder.pth https://huggingface.co/bizgonoman/kmd-model/resolve/main/vae_approx/taef1_encoder.pth

RUN comfy node install ComfyUI-Impact-Pack

  
# Stage 3: Final image
FROM base as final

# Copy models from stage 2 to the final image
COPY --from=downloader /comfyui/models /comfyui/models

# Start container
CMD ["/start.sh"]
