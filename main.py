import os
import pandas as pd
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from Src.Router import NutriSenseRouter
from Src.LLM.llm_engine import LLMEngine
from Src.LLM.llm_client import OllamaLLMClient

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'temp_uploads'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize NutriSense Core once
df = pd.read_csv("Dataset\processed\Final_unified_dataset.csv")
image_model = tf.keras.models.load_model("Src\Image_classifier\models\efficientb4_best.h5")
client = OllamaLLMClient()
engine = LLMEngine(client)
router = NutriSenseRouter(df, engine, image_model)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    text_query = request.form.get('query')
    image_file = request.files.get('image')
    
    image_path = None
    if image_file and image_file.filename != '':
        filename = secure_filename(image_file.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image_file.save(image_path)

    # Run AI logic
    result = router.execute(text_query=text_query, image_input=image_path)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)