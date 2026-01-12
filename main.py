import os
import pandas as pd
import tensorflow as tf
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from Src.Router.Router import NutriSenseRouter
from Src.LLM.llm_engine import LLMEngine
from Src.LLM.llm_client import OllamaLLMClient

# Initialize Flask app with custom template and static folders
app = Flask(__name__, 
            template_folder='Frontend',
            static_folder='Frontend')

app.config['UPLOAD_FOLDER'] = 'temp_uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize NutriSense Core once
print("Loading dataset...")
df = pd.read_csv(r"Dataset\processed\Final_unified_dataset.csv")
print(f"Dataset loaded: {len(df)} recipes")

print("Loading image model...")
image_model = tf.keras.models.load_model(r"Src\Image_classifier\models\efficientb4_best.h5")
print("Image model loaded successfully")

print("Initializing LLM engine...")
client = OllamaLLMClient()
engine = LLMEngine(client)
router = NutriSenseRouter(df, engine, image_model)
print("NutriSense AI initialized successfully!")

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
        
        # Validate input
        if not text_query and not image_file:
            return jsonify({
                'error': 'Please provide either a text query or an image'
            }), 400
        
        image_path = None
        if image_file and image_file.filename != '':
            # Validate file type
            allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
            file_ext = image_file.filename.rsplit('.', 1)[1].lower() if '.' in image_file.filename else ''
            
            if file_ext not in allowed_extensions:
                return jsonify({
                    'error': f'Invalid file type. Allowed: {", ".join(allowed_extensions)}'
                }), 400
            
            filename = secure_filename(image_file.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image_file.save(image_path)
            print(f"Image saved to: {image_path}")

        # Run AI logic
        print(f"Processing query: {text_query or 'Image only'}")
        result = router.execute(text_query=text_query if text_query else None, 
                               image_input=image_path)
        
        # Clean up uploaded image
        if image_path and os.path.exists(image_path):
            try:
                os.remove(image_path)
                print(f"Cleaned up: {image_path}")
            except Exception as e:
                print(f"Could not delete temp file: {e}")
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Error in /process: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': f'Server error: {str(e)}'
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'NutriSense AI is running',
        'recipes_loaded': len(df)
    })

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🥗 NutriSense AI Server Starting...")
    print("="*50)
    print(f"Frontend: http://localhost:5000")
    print(f"Health Check: http://localhost:5000/health")
    print("="*50 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)