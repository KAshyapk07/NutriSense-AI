import os
import json
import pandas as pd
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input 
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from Src.Router.Router import NutriSenseRouter
from Src.LLM.llm_engine import LLMEngine
from Src.LLM.llm_client import OllamaLLMClient

# Initialize Flask app
app = Flask(__name__, 
            template_folder='Frontend',
            static_folder='Frontend')

app.config['UPLOAD_FOLDER'] = 'temp_uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize NutriSense Core
print("Loading dataset...")
df = pd.read_csv(r"Dataset\processed\Final_unified_dataset.csv")
print(f"Dataset loaded: {len(df)} recipes")

print("Loading image model...")
image_model = tf.keras.models.load_model(r"Src\Image_classifier\models\efficientb4_best.h5")
print("Image model loaded successfully")

# LOAD CLASS NAMES

def load_class_names():
    """Load class names in the exact order the model was trained with."""
    
    # Option 1: Load from saved JSON file (recommended)
    json_path = "class_names.json"
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                class_names = json.load(f)
            print(f" Loaded {len(class_names)} classes from {json_path}")
            return class_names
        except Exception as e:
            print(f" Could not load {json_path}: {e}")
    
    # Option 2: Load from meta.json (if you saved it during training)
    meta_path = "meta.json"
    if os.path.exists(meta_path):
        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
                class_names = meta['class_names']
            print(f" Loaded {len(class_names)} classes from {meta_path}")
            return class_names
        except Exception as e:
            print(f" Could not load {meta_path}: {e}")
    
    # Option 3: Extract from dataset folder (fallback)
    dataset_path = r"Dataset\Images"
    
    if os.path.exists(dataset_path):
        try:
            class_folders = []
            for item in os.listdir(dataset_path):
                item_path = os.path.join(dataset_path, item)
                if os.path.isdir(item_path):
                    has_images = any(f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')) 
                                   for f in os.listdir(item_path))
                    if has_images:
                        class_folders.append(item)
            
            # Sort alphabetically (matches TensorFlow's behavior)
            class_names = sorted(class_folders)
            
            print(f" Extracted {len(class_names)} classes from dataset folder")
            
            # Save for next time
            try:
                with open(json_path, 'w') as f:
                    json.dump(class_names, f, indent=2)
                print(f" Saved class names to {json_path} for future use")
            except:
                pass
            
            return class_names
            
        except Exception as e:
            print(f" Could not extract from dataset: {e}")
    
    # Fallback
    print(" WARNING: Using generic class names. Predictions may be incorrect!")
    return [f"dish_{i}" for i in range(image_model.output_shape[1])]


CLASS_NAMES = load_class_names()

# Show first few classes
print(f"\n Class mapping (first 10):")
for idx in range(min(10, len(CLASS_NAMES))):
    print(f"   [{idx:3d}] {CLASS_NAMES[idx]}")
if len(CLASS_NAMES) > 10:
    print(f"   ... and {len(CLASS_NAMES) - 10} more")


class ImageModelWrapper:
    """
    Image model wrapper with EfficientNet preprocessing.
    CRITICAL: Must match the preprocessing used during training!
    """
    def __init__(self, model, class_names):
        self.model = model
        self.class_names = class_names
        self.img_size = (256, 256)
    
    def predict(self, image_path):
        """
        Predict dish from image using EfficientNet preprocessing.
        Returns: (dish_name: str, confidence: float)
        """
        try:
            # Load image
            img = Image.open(image_path)
            img = img.convert('RGB')
            img = img.resize(self.img_size)
            
            # Convert to numpy array
            img_array = np.array(img)
            img_array = np.expand_dims(img_array, axis=0)
            
            # CRITICAL: Apply EfficientNet preprocessing 
            # This normalizes the image to the range expected by EfficientNet
            img_array = preprocess_input(img_array)
            
            # Predict
            predictions = self.model.predict(img_array, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class])
            
            # Get dish name
            dish_name = self.class_names[predicted_class]
            
            # Log top 3 predictions for debugging
            print(f" Image prediction results:")
            top3_idx = np.argsort(predictions[0])[-3:][::-1]
            for idx in top3_idx:
                print(f"   [{idx:3d}] {self.class_names[idx]:30s} {predictions[0][idx]:6.2%}")
            
            return dish_name, confidence
            
        except Exception as e:
            print(f" Image prediction error: {e}")
            import traceback
            traceback.print_exc()
            return "unknown dish", 0.0


# Wrap the loaded model with correct class names
image_model_wrapper = ImageModelWrapper(image_model, CLASS_NAMES)

print("Initializing LLM engine...")
client = OllamaLLMClient()
engine = LLMEngine(client)
router = NutriSenseRouter(df, engine, image_model_wrapper)
print(" NutriSense AI initialized successfully!")


@app.route('/')
def index():
    """Serve the main frontend page"""
    return send_from_directory('Frontend', 'index.html')


@app.route('/<path:path>')
def serve_static(path):
    """Serve any static files from Frontend folder"""
    return send_from_directory('Frontend', path)


@app.route('/process', methods=['POST'])
def process():
    """Handle nutrition analysis requests"""
    try:
        text_query = request.form.get('query', '').strip()
        image_file = request.files.get('image')
        
        print(f" Received request - Query: '{text_query}', Image: {image_file.filename if image_file else 'None'}")
        
        if not text_query and not image_file:
            return jsonify({
                'error': 'Please provide either a text query or an image'
            }), 400
        
        image_path = None
        if image_file and image_file.filename != '':
            allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
            file_ext = image_file.filename.rsplit('.', 1)[1].lower() if '.' in image_file.filename else ''
            
            if file_ext not in allowed_extensions:
                return jsonify({
                    'error': f'Invalid file type. Allowed: {", ".join(allowed_extensions)}'
                }), 400
            
            filename = secure_filename(image_file.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image_file.save(image_path)
            print(f" Image saved to: {image_path}")

        print(f" Processing...")
        
        if image_path:
            result = router.execute(text_query=None, image_input=image_path)
        else:
            result = router.execute(text_query=text_query, image_input=None)
        
        # Clean up uploaded image
        if image_path and os.path.exists(image_path):
            try:
                os.remove(image_path)
                print(f" Cleaned up: {image_path}")
            except Exception as e:
                print(f" Could not delete temp file: {e}")
        
        print(f"Response ready: {result.get('pathway', 'unknown')} pathway")
        return jsonify(result)
    
    except Exception as e:
        print(f" Error in /process: {str(e)}")
        import traceback
        traceback.print_exc()
        
        if 'image_path' in locals() and image_path and os.path.exists(image_path):
            try:
                os.remove(image_path)
            except:
                pass
        
        return jsonify({
            'error': f'Server error: {str(e)}'
        }), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'NutriSense AI is running',
        'recipes_loaded': len(df),
        'image_model_loaded': router.image_model is not None,
        'num_classes': len(CLASS_NAMES),
        'sample_classes': CLASS_NAMES[:5]
    })


@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    print(" NutriSense AI Server Starting...")
    print("="*60)
    print(f" Frontend: http://localhost:5000")
    print(f" Health Check: http://localhost:5000/health")
    print(f" Image Model: {' Loaded' if router.image_model else ' Not loaded'}")
    print(f" Dish Classes: {len(CLASS_NAMES)}")
    print(f" Preprocessing: EfficientNet preprocess_input")

    app.run(debug=True, host='0.0.0.0', port=5000)