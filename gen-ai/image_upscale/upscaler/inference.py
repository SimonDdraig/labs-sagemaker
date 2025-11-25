# inference.py
# deploy a model container image to a sagemaker endpoint
# we do this because the upscaler uses a lot of gpu
# and rather than provisioning a huge instance, we can use a diffuser pipeline to split and batch the image
# but we need to direct this via a pipeline to cuda
# and by default doing this directs it to your notebooks cuda if you simply did this in a cell
# so we package up here and host it in a container on the provisioned instance
# so that it will direct to the provisioned instance on a sagemaker endpoint instead
#
# NOTE we need to ensure this is tarred up to create an inference script we can use when creating the model
# For example:
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
# If we just used the model via JumpStart, we'd likely get cuda errors (400) dues to large images or batch sizes and we can't control the use of the GPU

# we use a stability diffusion diffuser pipeline to batch the image being scaled and then stitch back together
# this allows us to use smaller instance sizes wher the GPU doesn't have to be huge to accommodate the source image in gpu memory
# https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler

from diffusers import StableDiffusionUpscalePipeline
from PIL import Image
import torch
import base64
import io
import json

# This loads the model once when the endpoint starts from stabilityai
# The "cuda" directive uses the SageMaker instance's GPU, not your local notebook GPU
def model_fn(model_dir):
    """Load model when the container starts."""
    print("=== STARTING model_fn ===")
    print(f"Model directory: {model_dir}")
    model_id = "stabilityai/stable-diffusion-x4-upscaler"
    print(f"Loading model: {model_id}")

    try:
        pipe = StableDiffusionUpscalePipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16
        )
        print("Model loaded successfully")

        pipe = pipe.to("cuda")
        print("Model moved to CUDA")

        pipe.safety_checker = None  # optional, skip NSFW model to save memory
        print("Disabled NSFW filters to save memory")

        pipe.enable_attention_slicing()  # optional: helps with memory
        print("Attention slicing enabled")

        # Log GPU memory info
        if torch.cuda.is_available():
            print(f"GPU Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            print(f"GPU Memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

        print("=== COMPLETED model_fn ===")
        return pipe
    except Exception as e:
        logger.error(f"Error in model_fn: {str(e)}")
        raise

# Converts the incoming base64 image back to a PIL Image that the model can process
def input_fn(request_body, content_type):
    """Deserialize input from JSON or raw base64 image."""
    print("=== STARTING input_fn ===")
    print(f"Content type: {content_type}")
    print(f"Request body length: {len(request_body)}")

    try:
        if content_type == "application/json":
            data = json.loads(request_body)
            prompt = data.get("prompt", "")
            image_b64 = data["image"]
            print(f"Prompt: {prompt}")
            print(f"Base64 image length: {len(image_b64)}")

            image_bytes = base64.b64decode(image_b64)
            print(f"Decoded image size: {len(image_bytes)} bytes")

            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            print(f"Image dimensions: {image.size}")

            print("=== COMPLETED input_fn ===")
            return {"prompt": prompt, "image": image}
        else:
            raise ValueError(f"Unsupported content type: {content_type}")
    except Exception as e:
        logger.error(f"Error in input_fn: {str(e)}")
        raise

# Runs the actual upscaling with memory-efficient inference mode
def predict_fn(input_data, model):
    """Run inference."""
    print("=== STARTING predict_fn ===")
    print(f"Input prompt: {input_data['prompt']}")
    print(f"Input image size: {input_data['image'].size}")

    try:
        with torch.inference_mode():
            # Log GPU memory before inference
            if torch.cuda.is_available():
                memory_before = torch.cuda.memory_allocated() / 1024**3
                print(f"GPU Memory before inference: {memory_before:.2f} GB")

            print("Starting model inference...")
            result = model(prompt=input_data["prompt"], image=input_data["image"])
            print(f"Result type: {type(result)}")
            print("Model inference completed successfully")

            # Log GPU memory after inference
            if torch.cuda.is_available():
                memory_after = torch.cuda.memory_allocated() / 1024**3
                print(f"GPU Memory after inference: {memory_after:.2f} GB")
                print(f"Memory delta: {memory_after - memory_before:.2f} GB")

            if hasattr(result, 'images'):
                print(f"Number of images generated: {len(result.images)}")

            # release GPU memory between requests
            torch.cuda.empty_cache()

            print("=== COMPLETED predict_fn ===")

            return result.images[0]
    except Exception as e:
        logger.error(f"Error in predict_fn: {str(e)}")
        # Log detailed CUDA memory info if available
        if torch.cuda.is_available():
            logger.error(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            logger.error(f"CUDA memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        raise

# Converts the upscaled image back to base64 for the HTTP response
def output_fn(prediction, accept):
    """Return base64-encoded PNG."""
    print("=== STARTING output_fn ===")
    print(f"Accept header: {accept}")
    print(f"Prediction type: {type(prediction)}")

    try:
        if hasattr(prediction, 'size'):
            print(f"Output image size: {prediction.size}")

        buf = io.BytesIO()
        prediction.save(buf, format="PNG")
        image_size = buf.getbuffer().nbytes
        print(f"PNG buffer size: {image_size} bytes")

        buf.seek(0)
        image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        print(f"Base64 output length: {len(image_b64)}")

        # allow async inference by writing outputs to /opt/ml/output (SageMaker picks that up)
        # ensures async inference always produces a valid output file
        result = {"upscaled_image": image_b64}
        out_path = "/opt/ml/output/result.json"
        with open(out_path, "w") as f:
            json.dump(result, f)
        print(f"Stored image for sagemaker to pickup at {out_path}")

        print("=== COMPLETED output_fn ===")
        return json.dumps(result)
    except Exception as e:
        logger.error(f"Error in output_fn: {str(e)}")
        raise