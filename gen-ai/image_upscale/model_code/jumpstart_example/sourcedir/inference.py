import base64
import json
import logging
import os
from typing import Any
from typing import Dict
from typing import Union

import numpy as np
from constants import constants
from diffusers import StableDiffusionUpscalePipeline
from PIL import Image
from sagemaker_inference import encoder
from six import BytesIO
from transformers import set_seed


def model_fn(model_dir: str) -> StableDiffusionUpscalePipeline:
    """Create our inference task as a delegate to the model.

    This runs only once per one worker.

    Args:
        model_dir (str): directory where the model files are stored
    Returns:
        StableDiffusionUpscalePipeline: a huggingface pipeline for generating image from text and image
    Raises:
        ValueError if the model file cannot be found.
    """
    try:
        model = StableDiffusionUpscalePipeline.from_pretrained(model_dir)
        model = model.to(constants.CUDA)
        model.enable_attention_slicing()
        return model
    except Exception:
        logging.exception(f"Failed to load model from: {model_dir}")
        raise


def _validate_payload(payload: Dict[str, Any]) -> None:
    """Validate the parameters in the input loads.

    Checks if num_inference_steps and num_images_per_prompt are integers.
    Checks if guidance_scale, num_return_sequences, num_beams, top_p and temprature are in bounds.
    Checks if do_sample is boolean.
    Checks max_length, num_return_sequences, num_beams and seed are integers.
    Args:
        payload: a decoded input payload (dictionary of input parameter and values)
    """

    for param_name in payload:
        assert (
            param_name in constants.ALL_PARAM_NAMES
        ), f"Input payload contains an invalid key {param_name}. Valid keys are {constants.ALL_PARAM_NAMES}."

    assert constants.PROMPT in payload, f"Input payload must contain '{constants.PROMPT}' key."
    assert constants.IMAGE in payload, f"Input payload must contain '{constants.IMAGE}' key."

    for param_name in [
        constants.NUM_INFERENCE_STEPS,
        constants.NUM_IMAGES_PER_PROMPT,
        constants.SEED,
        constants.NOISE_LEVEL,
    ]:
        if param_name in payload:
            assert type(payload[param_name]) == int, f"{param_name} must be an integer, got {payload[param_name]}."
    for param_name in [constants.GUIDANCE_SCALE, constants.ETA]:
        if param_name in payload:
            assert (
                type(payload[param_name]) == float or type(payload[param_name]) == int
            ), f"{param_name} must be an int or float, got {payload[param_name]}."
    if constants.NUM_INFERENCE_STEPS in payload:
        assert (
            payload[constants.NUM_INFERENCE_STEPS] >= 1
        ), f"{constants.NUM_INFERENCE_STEPS} must be at least 1, got {payload[constants.NUM_INFERENCE_STEPS]}."
    if constants.NUM_IMAGES_PER_PROMPT in payload:
        assert (
            payload[constants.NUM_IMAGES_PER_PROMPT] >= 1
        ), f"{constants.NUM_IMAGES_PER_PROMPT} must be at least 1, got {payload[constants.NUM_IMAGES_PER_PROMPT]}."


def encode_image_jpeg(image: Image.Image) -> str:
    """Encode the image in base64.b64 format after converting to JPEG format and loading as bytearray."""
    out = BytesIO()
    image.save(out, format=constants.JPEG_FORMAT)
    generated_image_bytes = out.getvalue()
    generated_image_encoded = base64.b64encode(generated_image_bytes).decode()
    return generated_image_encoded


def transform_fn(dreamer: StableDiffusionUpscalePipeline, input_data: bytes, content_type: str, accept: str) -> bytes:
    """Make predictions against the model and return a serialized response.

    The function signature conforms to the SM contract.

    Args:
        dreamer (StableDiffusionUpscalePipeline): a huggingface pipeline
        input_data (obj): the request data.
        content_type (str): the request content type.
        accept (str): accept header expected by the client.
    Returns:
        obj: a byte string of the prediction
    """
    if content_type in [constants.APPLICATION_JSON_JPEG, constants.APPLICATION_JSON]:
        try:
            payload = json.loads(input_data)
        except Exception:
            logging.exception(
                f"Failed to parse input payload. For content_type={constants.APPLICATION_JSON_JPEG}, input "
                f"payload must be a json encoded dictionary with keys {constants.ALL_PARAM_NAMES}."
            )
            raise
        _validate_payload(payload)
        if constants.SEED in payload:
            set_seed(payload[constants.SEED])
            del payload[constants.SEED]

        if content_type == constants.APPLICATION_JSON_JPEG:
            low_res_img_decoded = base64.b64decode(payload[constants.IMAGE].encode())
            with Image.open(BytesIO(low_res_img_decoded)) as f:
                low_res_img_rgb = f.convert("RGB")
                low_res_img_np_array = np.array(low_res_img_rgb)
        else:
            low_res_img_np_array = np.array(payload[constants.IMAGE], dtype="uint8")
        del payload[constants.IMAGE]

        with Image.fromarray(low_res_img_np_array) as low_res_img:
            generated_images = dreamer(image=low_res_img, **payload).images
        if constants.JPEG_ACCEPT_EXTENSION in accept:
            output = {
                constants.GENERATED_IMAGES: [encode_image_jpeg(image) for image in generated_images],
                constants.PROMPT: payload[constants.PROMPT],
            }
            accept = accept.replace(constants.JPEG_ACCEPT_EXTENSION, "")
        else:
            output = {
                constants.GENERATED_IMAGES: [np.asarray(generated_img) for generated_img in generated_images],
                constants.PROMPT: payload[constants.PROMPT],
            }
    else:
        raise ValueError('{{"error": "unsupported content type {}"}}'.format(content_type or "unknown"))

    if accept.endswith(constants.VERBOSE_EXTENSION):
        accept = accept.rstrip(constants.VERBOSE_EXTENSION)

    return encoder.encode(output, accept)
