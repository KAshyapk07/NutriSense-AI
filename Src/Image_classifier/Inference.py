# Src/Image_classifier/inference.py
import tensorflow as tf
import numpy as np
import json
import os
from tensorflow import keras
from keras.utils import load_img, img_to_array

def load_meta(meta_path):
    with open(meta_path, 'r') as f:
        return json.load(f)

def load_model_and_meta(model_path, meta_path):
    meta = load_meta(meta_path)
    # load model without compiling (faster if you only infer)
    model = tf.keras.models.load_model(model_path, compile=False)
    return model, meta

def preprocess_image(img_path, img_size):
    # uses tf.keras.utils.load_img / img_to_array
    img = load_img(img_path, target_size=(img_size, img_size))
    arr = img_to_array(img)  # shape (H,W,3)
    arr = np.expand_dims(arr, axis=0)  # (1,H,W,3)
    # apply model-specific preprocessing (EfficientNet expects preprocess_input)
    arr = tf.keras.applications.efficientnet.preprocess_input(arr)
    return arr

def predict_image(model, meta, img_path, top_k=1):
    img_size = meta.get('img_size', 380)
    class_names = meta['class_names']
    x = preprocess_image(img_path, img_size)
    preds = model.predict(x)
    top_indices = preds[0].argsort()[-top_k:][::-1]
    results = []
    for idx in top_indices:
        results.append({
            'label_index': int(idx),
            'label': class_names[idx],
            'score': float(preds[0][idx])
        })
    return results

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', required=True)
    p.add_argument('--meta_path', required=True)
    p.add_argument('--img', required=True)
    p.add_argument('--top_k', type=int, default=1)
    args = p.parse_args()

    model, meta = load_model_and_meta(args.model_path, args.meta_path)
    res = predict_image(model, meta, args.img, top_k=args.top_k)
    print(res)
