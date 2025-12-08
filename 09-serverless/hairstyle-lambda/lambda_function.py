import json
from io import BytesIO
from urllib import request

import numpy as np
from PIL import Image
import onnxruntime as ort

MODEL_PATH = "hair_classifier_empty.onnx" 

def download_image(url: str) -> Image.Image:
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img

def prepare_image(img: Image.Image, target_size=(200, 200)) -> Image.Image:
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

def preprocess(img: Image.Image) -> np.ndarray:
    x = np.array(img).astype("float32")  # (H, W, C), valores 0..255
    x = x / 255.0                        # [0, 1]

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    x = (x - mean) / std                 # ImageNet normalization
    x = np.transpose(x, (2, 0, 1))       # (C, H, W)
    x = np.expand_dims(x, axis=0)        # (1, C, H, W)

    return x


session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

def lambda_handler(event, context=None):
    url = event["url"]

    img = download_image(url)
    img = prepare_image(img, (200, 200))
    x = preprocess(img)

    pred = session.run([output_name], {input_name: x})
    score = float(pred[0][0][0])

    return {
        "statusCode": 200,
        "body": json.dumps({"score": score})
    }