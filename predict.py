# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os

MODEL_CACHE = "model_cache"

# Set environment variables for model caching BEFORE any imports
os.environ["HF_HOME"] = MODEL_CACHE
os.environ["TORCH_HOME"] = MODEL_CACHE
os.environ["HF_DATASETS_CACHE"] = MODEL_CACHE
os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = MODEL_CACHE

import subprocess
import time
from typing import List, Optional

import torch
from cog import BaseModel, BasePredictor, File, Input
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, CLIPModel

from helpers import record_billing_metric

MODEL_NAME = "openai/clip-vit-large-patch14"
BASE_URL = "https://weights.replicate.delivery/default/clip-embeddings/model_cache/"

device = "cuda" if torch.cuda.is_available() else "cpu"


def download_weights(url: str, dest: str) -> None:
    start = time.time()
    print("[!] Initiating download from URL: ", url)
    print("[~] Destination path: ", dest)
    if ".tar" in dest:
        dest = os.path.dirname(dest)
    command = ["pget", "-vf" + ("x" if ".tar" in url else ""), url, dest]
    try:
        print(f"[~] Running command: {' '.join(command)}")
        subprocess.check_call(command, close_fds=False)
    except subprocess.CalledProcessError as e:
        print(
            f"[ERROR] Failed to download weights. Command '{' '.join(e.cmd)}' returned non-zero exit status {e.returncode}."
        )
        raise
    print("[+] Download completed in: ", time.time() - start, "seconds")


class Output(BaseModel):
    embedding: List[float]


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # Create model cache directory if it doesn't exist
        os.makedirs(MODEL_CACHE, exist_ok=True)

        # Download model weights if they don't exist
        model_file = "models--openai--clip-vit-large-patch14.tar"
        url = f"{BASE_URL}{model_file}"
        dest_path = os.path.join(MODEL_CACHE, model_file)
        extracted_path = dest_path.replace(".tar", "")

        if not os.path.exists(extracted_path):
            print(f"[~] Downloading {model_file}...")
            download_weights(url, dest_path)
        else:
            print(f"[✓] {model_file} already exists, skipping download")

        # Load the model using the cache
        self.model: CLIPModel = CLIPModel.from_pretrained(
            MODEL_NAME, cache_dir=MODEL_CACHE
        )
        self.model = self.model.to(device)
        self.model.eval()  # Set to evaluation mode

        self.processor = AutoProcessor.from_pretrained(
            MODEL_NAME, cache_dir=MODEL_CACHE
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, cache_dir=MODEL_CACHE
        )

    def predict(
        self,
        text: Optional[str] = Input(description="Input text to encode", default=None),
        image: Optional[File] = Input(
            description="Input image to encode", default=None
        ),
    ) -> Output:
        """Run a single prediction on the model

        Provide either text or image input (not both). If both are provided,
        only the image will be processed.
        """
        
        # Start timing for billing
        start_time = time.time()
        
        embedding = []

        if image is not None:
            pil_image = Image.open(image)
            inputs = self.processor(images=pil_image, return_tensors="pt").to(device)
            image_features = self.model.get_image_features(**inputs)
            embedding = image_features.tolist()[0]

        elif text is not None:
            inputs = self.tokenizer([text], padding=True, return_tensors="pt").to(
                device
            )
            text_features = self.model.get_text_features(**inputs)
            embedding = text_features.tolist()[0]

        # Calculate elapsed time and record time-based billing metric
        elapsed_time = time.time() - start_time
        record_billing_metric("unspecified_billing_metric", elapsed_time)

        return Output(embedding=embedding)
