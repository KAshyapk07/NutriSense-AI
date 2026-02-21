from __future__ import annotations

import json
import logging
import os
from typing import Optional

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input

logger = logging.getLogger(__name__)

MODEL_PATH = os.getenv(
    "MODEL_PATH", r"Src\Image_classifier\models\efficientb4_best.h5"
)
CLASS_NAMES_PATH = os.getenv("CLASS_NAMES_PATH", "class_names.json")
IMAGE_SIZE = (256, 256)


class ImageModelWrapper:
    def __init__(self, model: tf.keras.Model, class_names: list[str]):
        self._model = model
        self._class_names = class_names

    def predict(self, image_path: str) -> tuple[str, float]:
        img = Image.open(image_path).convert("RGB").resize(IMAGE_SIZE)
        img_array = preprocess_input(
            np.expand_dims(np.array(img), axis=0).astype("float32")
        )
        predictions = self._model.predict(img_array, verbose=0)
        idx = int(np.argmax(predictions[0]))
        return self._class_names[idx], float(predictions[0][idx])

    @property
    def num_classes(self) -> int:
        return len(self._class_names)

    @property
    def class_names(self) -> list[str]:
        return self._class_names


_wrapper: Optional[ImageModelWrapper] = None


def _load_class_names() -> list[str]:
    """Three-tier fallback matching existing main.py logic."""
    # Tier 1 — class_names.json at project root
    if os.path.exists(CLASS_NAMES_PATH):
        with open(CLASS_NAMES_PATH, "r") as f:
            return json.load(f)

    # Tier 2 — meta.json alongside the model
    meta_path = os.path.join(os.path.dirname(MODEL_PATH), "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
            if "class_names" in meta:
                return meta["class_names"]

    # Tier 3 — scan Dataset/Images/ folder structure
    images_dir = "Dataset/Images"
    if os.path.isdir(images_dir):
        names = sorted(
            d
            for d in os.listdir(images_dir)
            if os.path.isdir(os.path.join(images_dir, d))
        )
        if names:
            logger.warning(
                "class_names.json not found; derived %d classes from Dataset/Images/",
                len(names),
            )
            return names

    raise FileNotFoundError(
        "Could not load class names from class_names.json, meta.json, "
        "or Dataset/Images/."
    )


def init() -> None:
    global _wrapper
    model = tf.keras.models.load_model(MODEL_PATH)
    class_names = _load_class_names()
    _wrapper = ImageModelWrapper(model, class_names)
    logger.info(
        "Image model loaded — %d classes | path: %s",
        _wrapper.num_classes,
        MODEL_PATH,
    )


def get_image_model() -> ImageModelWrapper:
    if _wrapper is None:
        raise RuntimeError("Image model is not initialized.")
    return _wrapper
