import os
import json
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from Src.Router.Router import NutriSenseRouter
from Src.LLM.llm_engine import LLMEngine
from Src.LLM.llm_client import OllamaLLMClient
from Src.neo4j_client import Neo4jClient

app = Flask(__name__, template_folder='Frontend', static_folder='Frontend')
app.config['UPLOAD_FOLDER'] = 'temp_uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

neo4j_client = Neo4jClient()
stats = neo4j_client.get_stats()
print(f"Neo4j: {stats['recipes']} recipes | {stats['cuisines']} cuisines")

image_model = tf.keras.models.load_model(r"Src\Image_classifier\models\efficientb4_best.h5")


def load_class_names():
    json_path = "class_names.json"
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                return json.load(f)
        except Exception:
            pass

    meta_path = "meta.json"
    if os.path.exists(meta_path):
        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
                return meta['class_names']
        except Exception:
            pass

    dataset_path = r"Dataset\Images"
    if os.path.exists(dataset_path):
        try:
            class_folders = [
                item for item in os.listdir(dataset_path)
                if os.path.isdir(os.path.join(dataset_path, item))
                and any(
                    f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))
                    for f in os.listdir(os.path.join(dataset_path, item))
                )
            ]
            class_names = sorted(class_folders)
            try:
                with open(json_path, 'w') as f:
                    json.dump(class_names, f, indent=2)
            except Exception:
                pass
            return class_names
        except Exception:
            pass

    return [f"dish_{i}" for i in range(image_model.output_shape[1])]


CLASS_NAMES = load_class_names()


class ImageModelWrapper:
    def __init__(self, model, class_names):
        self.model = model
        self.class_names = class_names
        self.img_size = (256, 256)

    def predict(self, image_path):
        try:
            img = Image.open(image_path).convert('RGB').resize(self.img_size)
            img_array = preprocess_input(np.expand_dims(np.array(img), axis=0))
            predictions = self.model.predict(img_array, verbose=0)
            predicted_class = np.argmax(predictions[0])
            return self.class_names[predicted_class], float(predictions[0][predicted_class])
        except Exception as e:
            import traceback
            traceback.print_exc()
            return "unknown dish", 0.0


image_model_wrapper = ImageModelWrapper(image_model, CLASS_NAMES)
client = OllamaLLMClient()
engine = LLMEngine(client)
router = NutriSenseRouter(neo4j_client, engine, image_model_wrapper)


@app.route('/')
def index():
    return send_from_directory('Frontend', 'index.html')


@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('Frontend', path)


@app.route('/process', methods=['POST'])
def process():
    try:
        text_query = request.form.get('query', '').strip()
        image_file = request.files.get('image')

        if not text_query and not image_file:
            return jsonify({'error': 'Please provide either a text query or an image'}), 400

        image_path = None
        if image_file and image_file.filename != '':
            allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
            file_ext = image_file.filename.rsplit('.', 1)[1].lower() if '.' in image_file.filename else ''
            if file_ext not in allowed_extensions:
                return jsonify({'error': f'Invalid file type. Allowed: {", ".join(allowed_extensions)}'}), 400
            filename = secure_filename(image_file.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image_file.save(image_path)

        result = router.execute(
            text_query=None if image_path else text_query,
            image_input=image_path
        )

        if image_path and os.path.exists(image_path):
            try:
                os.remove(image_path)
            except Exception:
                pass

        return jsonify(result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        if 'image_path' in locals() and image_path and os.path.exists(image_path):
            try:
                os.remove(image_path)
            except Exception:
                pass
        return jsonify({'error': f'Server error: {str(e)}'}), 500


@app.route('/health', methods=['GET'])
def health_check():
    stats = neo4j_client.get_stats()
    return jsonify({
        'status': 'healthy',
        'recipes_loaded': stats['recipes'],
        'ingredients': stats['ingredients'],
        'cuisines': stats['cuisines'],
        'image_model_loaded': router.image_model is not None,
        'num_classes': len(CLASS_NAMES),
        'data_source': 'Neo4j'
    })


@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)