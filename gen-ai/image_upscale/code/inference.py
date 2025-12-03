# inference.py
# deploy a model container image to a sagemaker endpoint
# we do this because the upscaler uses a lot of gpu
# and rather than provisioning a huge instance, we can use a diffuser pipeline to split and batch the image
# but we need to direct this via a pipeline to cuda
# so we package up here and host it in a container on the provisioned instance
#
# NOTE we need to ensure this is tarred up to create an inference script we can use when creating the model
# To create the tar file, execute the following in your terminal
# tar -czvf model.tar.gz inference.py requirements.txt
#
# NOTE the model id below
# we cant use the jumpstart model id model-upscaling-stabilityai-stable-diffusion-x4-upscaler-fp16 because it has its own
# prebuilt container image and predefined model artifacts
# so we use one directly from stabilityai instead stabilityai/stable-diffusion-x4-upscaler so we can customise how the pipeline runs internally

# we import StableDiffusionUpscalePipeline to:
# Control how input and output are serialized (JSON, base64, etc.).
# Adjust memory settings (enable_attention_slicing, enable_model_cpu_offload).
# Mix multiple models into one endpoint.
# Integrate preprocessing/postprocessing (e.g., crop, resize, watermark).
# Use the latest Hugging Face weights directly, not the static JumpStart snapshot.
# If we just used the model via JumpStart, we'd likely get cuda errors (400) due to large images or batch sizes and we can't control the use of the GPU

# we use a stability diffusion diffuser pipeline to batch the image being scaled and then stitch back together
# this allows us to use smaller instance sizes where the GPU doesn't have to be huge to accommodate the source image in gpu memory
# https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler

from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_upscale import StableDiffusionUpscalePipeline
from PIL import Image
import torch
import base64
from io import BytesIO
import json
import os

def model_fn(model_dir):
    pipeline = StableDiffusionUpscalePipeline.from_pretrained(
        model_dir,
        torch_dtype=torch.float16
    ).to("cuda")
    # Memory optimizations
    pipeline.enable_attention_slicing()
    # Uncomment below for small GPU instances
    # pipeline.enable_model_cpu_offload()
    pipeline.eval()
    
    return pipeline

def input_fn(request_body, content_type):
    payload = json.loads(request_body)
    prompt = payload.get("prompt", "")
    image_b64 = payload["image"]

    img_bytes = base64.b64decode(image_b64)
    img = Image.open(BytesIO(img_bytes)).convert("RGB")

    return {"prompt": prompt, "image": img}

def predict_fn(inputs, model):
    prompt = inputs["prompt"]
    image = inputs["image"]

    result = model(prompt=prompt, image=image)
    out = result.images[0]

    buffer = BytesIO()
    out.save(buffer, format="PNG")
    out_b64 = base64.b64encode(buffer.getvalue()).decode()

    return {"image": out_b64}

def output_fn(prediction, accept):
    return json.dumps(prediction), accept
