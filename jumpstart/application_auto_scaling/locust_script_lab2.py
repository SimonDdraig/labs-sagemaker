import json
import logging
import os
import time
import random

import boto3
from botocore.config import Config
from locust.contrib.fasthttp import FastHttpUser

from locust import task, events

# get the vars exported in the terminal before running locust
region = os.environ["AWS_REGION"]
contentType = os.environ["CONTENT_TYPE"]
payload = os.environ["PAYLOAD"]
endpointName = os.environ['ENDPOINT_NAME']
modelType = os.environ['MODEL_TYPE']
 
class BotoClient:
    def __init__(self, host):
        if modelType == "llm":
            config = Config(
                region_name=region, retries={"max_attempts": 0, "mode": "standard"}
            )
        else:
            # increase timeout for image models
            config = Config(
                region_name=region,
                retries={"max_attempts": 0, "mode": "standard"},
                read_timeout=60,  # 1 minute
                connect_timeout=60
            )

        self.sagemaker_client = boto3.client("sagemaker-runtime", config=config)
        self.endpoint_name = endpointName
        self.content_type = contentType

        if modelType == "llm":
            self.payload = payload
        else:
            # for image models we need to create a json payload with additional parameters
            # may need to change these per image model - default is good for stable diffusion
            payload_dict = {
                "text_prompts": [
                    {"text": payload}
                ],
                "width": 128,
                "height": 128,
                "cfg_scale": 7.0,
                "steps": 50,
                "seed": random.randint(0, 4294967295)
            }
            self.payload = json.dumps(payload_dict).encode("utf-8")

    def send(self):
        from PIL import Image
        import base64
        from io import BytesIO

        request_meta = {
            "request_type": "InvokeEndpoint",
            "name": "SageMaker",
            "start_time": time.time(),
            "response_length": 0,
            "response": None,
            "context": {},
            "exception": None,
        }
        start_perf_counter = time.perf_counter()

        try:
            response = self.sagemaker_client.invoke_endpoint(
                EndpointName=self.endpoint_name,
                Body=self.payload,
                ContentType=self.content_type,
            )
            if modelType == "llm":
                logging.info(response["Body"].read())
            else:
                logging.info("Image generated")
                result = json.loads(response["Body"].read())
                image_b64 = result["generated_images"][0]
                image_bytes = base64.b64decode(image_b64)
                image = Image.open(BytesIO(image_bytes))
                image.show()
                image.close()

        except Exception as e:
            request_meta["exception"] = e

        request_meta["response_time"] = (
            time.perf_counter() - start_perf_counter
        ) * 1000

        events.request.fire(**request_meta)


class BotoUser(FastHttpUser):
    abstract = True

    def __init__(self, env):
        super().__init__(env)
        self.client = BotoClient(self.host)

# locust will look for a class that inherits from FastHttpUser or HttpUser and use it as the simulated user
class MyUser(BotoUser):
    @task
    def send_request(self):
        self.client.send()