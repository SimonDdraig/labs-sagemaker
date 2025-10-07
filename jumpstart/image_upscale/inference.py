# inference.py
# deploy a model container image to a sagemaker endpoint
# we do this because the upscaler uses a lot of gpu
# and rather than provisioning a huge isntance, we can use a diffuser to split and batch the image
# but we need to direct this via a pipeline to cuda
# and by default doing this directs it to your notebooks cuda
# so we package up here so that it will direct to the provisioned instance on a sagemaker endpoint instead
#
# NOTE we need to ensure this is zipped up to create an inference script we can use when creating the model
# For example:
# tar -czvf model.tar.gz inference.py
#
# NOTE the model id below
# we cant use the jumpstart model id model-upscaling-stabilityai-stable-diffusion-x4-upscaler-fp16 because it has its own
# prebuilt container image and predefined model artifacts
# so we use one directly from huggingface instead stabilityai/stable-diffusion-x4-upscaler so we can customise how the pipeline runs internally

# we import StableDiffusionUpscalePipeline to:
# Control how input and output are serialized (JSON, base64, etc.).
# Adjust memory settings (enable_attention_slicing, enable_model_cpu_offload).
# Mix multiple models into one endpoint.
# Integrate preprocessing/postprocessing (e.g., crop, resize, watermark).
# Use the latest Hugging Face weights directly, not the static JumpStart snapshot.
# If we just used the model via JumpStart, we'd likely get cuda errors (400) dues to large images or batch sizes and we can't control the use of the GPU

# we use a stability diffusion diffuser to batch the image being scaled and then stitch back together
# this allows us to use smaller instance sizes wher the GPU doesn't have to be huge to accommodate the source image in memory
# https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler

from diffusers import StableDiffusionUpscalePipeline
from PIL import Image
import torch
import base64
import io
import json

def model_fn(model_dir):
    """Load model when the container starts."""
    model_id = "stabilityai/stable-diffusion-x4-upscaler"
    pipe = StableDiffusionUpscalePipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")
    pipe.enable_attention_slicing()  # optional: helps with memory
    return pipe

def input_fn(request_body, content_type):
    """Deserialize input from JSON or raw base64 image."""
    if content_type == "application/json":
        data = json.loads(request_body)
        prompt = data.get("prompt", "")
        image_b64 = data["image"]
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return {"prompt": prompt, "image": image}
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(input_data, model):
    """Run inference."""
    with torch.inference_mode():
        result = model(prompt=input_data["prompt"], image=input_data["image"])
    return result.images[0]

def output_fn(prediction, accept):
    """Return base64-encoded PNG."""
    buf = io.BytesIO()
    prediction.save(buf, format="PNG")
    buf.seek(0)
    image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return json.dumps({"upscaled_image": image_b64})
