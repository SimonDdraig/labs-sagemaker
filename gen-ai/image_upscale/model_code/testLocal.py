import json
import base64
from PIL import Image
from io import BytesIO
from inference import model_fn, input_fn, predict_fn, output_fn

# 1. Load the model from your local model directory
model_dir = "./model"  # path to the folder containing the pretrained pipeline files
model = model_fn(model_dir)

# 2. Load an image to test
image_path = "resources/img2_original_512.jpeg"
with open(image_path, "rb") as f:
    img_bytes = f.read()
image_b64 = base64.b64encode(img_bytes).decode("utf-8")

# 3. Build a dummy payload
payload = {
    "prompt": "highly detailed, realistic photo",
    "image": image_b64
}
request_body = json.dumps(payload)

# 4. Call input_fn
inputs = input_fn(request_body, content_type="application/json")

# 5. Call predict_fn
prediction = predict_fn(inputs, model)

# 6. Call output_fn
response, content_type = output_fn(prediction, accept="application/json")

# 7. Show result
response_dict = json.loads(response)
upscaled_img_b64 = response_dict["image"]
upscaled_img_bytes = base64.b64decode(upscaled_img_b64)
upscaled_img = Image.open(BytesIO(upscaled_img_bytes))
upscaled_img.show()
