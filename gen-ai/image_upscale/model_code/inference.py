# inference.py
# deploy a model container image to a sagemaker endpoint
# we do this because the upscaler uses a lot of gpu
# and rather than provisioning a huge instance, we can use a diffuser pipeline to split and batch the image
# but we need to direct this via a pipeline to cuda to do this
# so we package up here and host it in a container on the provisioned endpoint
#
# NOTE we need to ensure this is tarred up to create an inference script we can use when creating the model
# To create the tar file, execute the following in your terminal
# tar -czvf model.tar.gz inference.py requirements.txt - the lab will do this for you
#
# NOTE the model id below
# we cant use the jumpstart model id model-upscaling-stabilityai-stable-diffusion-x4-upscaler-fp16 because it has its own
# prebuilt container image and predefined model artifacts
# so we use one directly from stabilityai via huggingface instead stabilityai/stable-diffusion-x4-upscaler so we can customise how the pipeline runs internally

# we import StableDiffusionUpscalePipeline so we can do the following if we wanted to customise further:
#     Control how input and output are serialized (JSON, base64, etc.).
#     Adjust memory settings (enable_attention_slicing, enable_model_cpu_offload).
#     Mix multiple models into one endpoint - we wont do that here.
#     Integrate preprocessing/postprocessing (e.g., crop, resize, watermark).
#     Use the latest Hugging Face weights directly, not the static JumpStart snapshot.
#     If we just used the model via JumpStart, we'd likely get cuda errors (400) due to large images or batch sizes and we can't control the use of the GPU

# we use a stability diffusion diffuser pipeline to batch the image being scaled and then stitch back together
# this allows us to use smaller instance sizes where the GPU doesn't have to be huge to accommodate the source image in gpu memory
# https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler

from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_upscale import StableDiffusionUpscalePipeline
from PIL import Image
import torch
import base64
from io import BytesIO
import json
import logging
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.info("inference.py loaded")

def model_fn(model_dir):
    logger.info("=============================================================================")
    model_id = "stabilityai/stable-diffusion-x4-upscaler"
    logger.info(f"model_fn Loading model {model_id} from {model_dir}")
    logger.info("=============================================================================")

    # accommodate for local testing, not everyone has a gpu on their desk!
    # for local testing, if you dont have a gpu, but have a mac it will use mps still blow your memory however
    # highly recommend use testLocal.py before pushing this to AWS
    # uncomment the mps lines if you have a mac with an M1/M2 chip and want to test locally with that, default to cpu if it blows memory
    #device = (
    #    "cuda" if torch.cuda.is_available()
    #    else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    #    else "cpu"
    #)
    logger.info("=============================================================================")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"model_fn using device: {device}")
    torchf = torch.float16 if device == "cuda" else torch.float32
    logger.info(f"model_fn using torch dtype: {torchf}")
    logger.info("=============================================================================")

    # create the pipeline
    pipeline = StableDiffusionUpscalePipeline.from_pretrained(
        model_dir,
        torch_dtype=torchf
    ).to(device)
    # Memory optimizations
    logger.info("=============================================================================")
    logger.info(f"model_fn enable_attention_slicing")
    pipeline.enable_attention_slicing()
    logger.info("=============================================================================")
    # Uncomment below for small GPU instances
    # pipeline.enable_model_cpu_offload()
    
    return pipeline

def input_fn(request_body, content_type):
    logger.info("=============================================================================")
    logger.info(f"input_fn request_body (logging 1st 100 chars only): {request_body[:100]}")
    logger.info(f"input_fn content_type: {content_type}")
    logger.info("=============================================================================")

    payload = json.loads(request_body)
    prompt = payload.get("prompt", "")
    image_b64 = payload["image"]

    img_bytes = base64.b64decode(image_b64)
    img = Image.open(BytesIO(img_bytes)).convert("RGB")

    return {"prompt": prompt, "image": img}

def predict_fn(inputs, model):
    logger.info("=============================================================================")
    logger.info(f"predict_fn inputs: {inputs}")
    logger.info("=============================================================================")

    prompt = inputs["prompt"]
    image = inputs["image"]

    logger.info("=============================================================================")
    logger.info(f"predict_fn invoking model")
    logger.info("=============================================================================")
    result = model(prompt=prompt, image=image)
    out = result.images[0]

    buffer = BytesIO()
    out.save(buffer, format="PNG")
    out_b64 = base64.b64encode(buffer.getvalue()).decode()

    logger.info("=============================================================================")
    logger.info(f"predict_fn returning image (logging 1st 100 chars only): {out_b64[:100]}")
    logger.info("=============================================================================")
    return {"image": out_b64}

def output_fn(prediction, accept):
    logger.info("=============================================================================")
    logger.info(f"output_fn accept: {accept}")
    accept = accept or "application/json"   
    logger.info("=============================================================================")
    return json.dumps(prediction), accept
