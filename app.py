from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import rembg
import os
import uuid
import logging
import base64
from werkzeug.utils import secure_filename
from io import BytesIO

app = Flask(__name__)
# Configure CORS with specific settings
CORS(app, resources={
    r"/*": {
        "origins": ["http://127.0.0.1:5500", "http://localhost:5500", "http://127.0.0.1:5000"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

# Create required directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/remove-bg', methods=['POST'])
def remove_background():
    try:
        logger.debug("Received request for background removal")
        
        if 'image' not in request.files:
            logger.error("No image file in request")
            return jsonify({'success': False, 'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            logger.error("Empty filename")
            return jsonify({'success': False, 'error': 'No image selected'}), 400
        
        if not allowed_file(file.filename):
            logger.error(f"Invalid file type: {file.filename}")
            return jsonify({'success': False, 'error': 'Invalid file type. Allowed types: PNG, JPG, JPEG, GIF, WebP, BMP'}), 400
        
        # Generate unique filename for temporary storage
        filename = str(uuid.uuid4()) + os.path.splitext(secure_filename(file.filename))[1]
        input_path = os.path.join(UPLOAD_FOLDER, filename)
        
        # Save uploaded file temporarily
        file.save(input_path)
        
        try:
            logger.debug("Processing image with rembg")
            # Process with rembg
            with open(input_path, 'rb') as f:
                img_data = f.read()
                output_data = rembg.remove(img_data)
            
            # Convert the processed image to base64
            base64_image = base64.b64encode(output_data).decode('utf-8')
            
            response = jsonify({
                'success': True,
                'processed_image': f'data:image/png;base64,{base64_image}'
            })
            
            # Add CORS headers explicitly
            response.headers.add('Access-Control-Allow-Origin', '*')
            response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
            response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            raise
        finally:
            # Clean up input file
            if os.path.exists(input_path):
                logger.debug(f"Cleaning up input file: {input_path}")
                os.remove(input_path)
    
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        # Clean up any files in case of error
        if 'input_path' in locals() and os.path.exists(input_path):
            os.remove(input_path)
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)