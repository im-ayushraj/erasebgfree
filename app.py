from flask import Flask, request, jsonify, render_template, send_file, send_from_directory
from flask_cors import CORS
import rembg
import os
import uuid
import logging
import base64
from werkzeug.utils import secure_filename
from io import BytesIO
from PIL import Image, ImageEnhance, ImageOps, ImageFilter, ImageDraw, ImageFont
from PIL.Image import LANCZOS
import numpy as np
from PyPDF2 import PdfReader, PdfWriter
from PyPDF2.generic import NameObject
from pdf2image import convert_from_bytes
import tempfile
import shutil
import subprocess
from pdf2docx import Converter
import pdfplumber
import openpyxl
from pdfminer.high_level import extract_text_to_fp



app = Flask(__name__)
# Configure CORS with specific settings
CORS(app, resources={
    r"/*": {
        "origins": ["http://127.0.0.1:5000", "http://localhost:5000"],
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

def allowed_pdf(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'pdf'

def compress_image(image_bytes, quality=70):
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()

def resize_image(image_bytes, width, height):
    img = Image.open(BytesIO(image_bytes))
    try:
        resample = Image.Resampling.LANCZOS
    except AttributeError:
        resample = LANCZOS
    img = img.resize((width, height), resample)
    buf = BytesIO()
    img.save(buf, format=img.format if img.format else "PNG")
    return buf.getvalue()
    return buf.getvalue()

def convert_image(image_bytes, fmt):
    img = Image.open(BytesIO(image_bytes))
    buf = BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()

def rotate_image(image_bytes, angle):
    img = Image.open(BytesIO(image_bytes))
    img = img.rotate(angle, expand=True)
    buf = BytesIO()
    img.save(buf, format=img.format if img.format else "PNG")
    return buf.getvalue()

def flip_image(image_bytes, mode):
    img = Image.open(BytesIO(image_bytes))
    try:
        FLIP_LEFT_RIGHT = Image.Transpose.FLIP_LEFT_RIGHT
        FLIP_TOP_BOTTOM = Image.Transpose.FLIP_TOP_BOTTOM
    except AttributeError:
        FLIP_LEFT_RIGHT = Image.FLIP_LEFT_RIGHT
        FLIP_TOP_BOTTOM = Image.FLIP_TOP_BOTTOM
    if mode == "horizontal":
        img = img.transpose(FLIP_LEFT_RIGHT)
    elif mode == "vertical":
        img = img.transpose(FLIP_TOP_BOTTOM)
    buf = BytesIO()
    img.save(buf, format=img.format if img.format else "PNG")
    return buf.getvalue()

def apply_filter(image_bytes, filter_type):
    img = Image.open(BytesIO(image_bytes))
    if filter_type == "grayscale":
        img = ImageOps.grayscale(img)
    elif filter_type == "invert":
        img = ImageOps.invert(img.convert("RGB"))
    elif filter_type == "blur":
        img = img.filter(ImageFilter.GaussianBlur(3))
    elif filter_type == "contour":
        img = img.filter(ImageFilter.CONTOUR)
    elif filter_type == "sharpen":
        img = img.filter(ImageFilter.SHARPEN)
    elif filter_type == "sepia":
        sepia = np.array(img.convert("RGB"))
        tr = [int((r * .393) + (g *.769) + (b * .189)) for r,g,b in sepia.reshape(-1,3)]
        tg = [int((r * .349) + (g *.686) + (b * .168)) for r,g,b in sepia.reshape(-1,3)]
        tb = [int((r * .272) + (g *.534) + (b * .131)) for r,g,b in sepia.reshape(-1,3)]
        sepia[...,0] = np.clip(tr, 0, 255)
        sepia[...,1] = np.clip(tg, 0, 255)
        sepia[...,2] = np.clip(tb, 0, 255)
        img = Image.fromarray(sepia.astype('uint8'))
    elif filter_type == "posterize":
        img = ImageOps.posterize(img.convert("RGB"), 3)
    elif filter_type == "solarize":
        img = ImageOps.solarize(img.convert("RGB"), threshold=128)
    buf = BytesIO()
    img.save(buf, format=img.format if img.format else "PNG")
    return buf.getvalue()

def adjust_image(image_bytes, brightness=1.0, contrast=1.0, color=1.0):
    img = Image.open(BytesIO(image_bytes))
    img = ImageEnhance.Brightness(img).enhance(brightness)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    img = ImageEnhance.Color(img).enhance(color)
    buf = BytesIO()
    img.save(buf, format=img.format if img.format else "PNG")
    return buf.getvalue()

def crop_image(image_bytes, x, y, width, height):
    img = Image.open(BytesIO(image_bytes))
    cropped = img.crop((x, y, x + width, y + height))
    buf = BytesIO()
    cropped.save(buf, format=img.format if img.format else "PNG")
    return buf.getvalue()

def add_border(image_bytes, color, thickness):
    img = Image.open(BytesIO(image_bytes))
    thickness = int(thickness)
    color = color.lstrip('#')
    color = tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
    new_size = (img.width + 2*thickness, img.height + 2*thickness)
    bordered = Image.new("RGB", new_size, color)
    bordered.paste(img, (thickness, thickness))
    buf = BytesIO()
    bordered.save(buf, format=img.format if img.format else "PNG")
    return buf.getvalue()

def rounded_corners(image_bytes, radius):
    img = Image.open(BytesIO(image_bytes)).convert("RGBA")
    radius = int(radius)
    mask = Image.new('L', img.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle([0, 0, img.size[0], img.size[1]], radius, fill=255)
    img.putalpha(mask)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def add_watermark(image_bytes, text, position, opacity):
    img = Image.open(BytesIO(image_bytes)).convert("RGBA")
    watermark = Image.new("RGBA", img.size)
    draw = ImageDraw.Draw(watermark)
    font_size = int(min(img.size) / 12)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
    # Use textbbox for Pillow >= 8.0, else fallback to textsize
    if hasattr(draw, "textbbox"):
        bbox = draw.textbbox((0, 0), text, font=font)
        textwidth = bbox[2] - bbox[0]
        textheight = bbox[3] - bbox[1]
    else:
        textwidth, textheight = font.getsize(text)
    positions = {
        "bottom-right": (img.width - textwidth - 10, img.height - textheight - 10),
        "bottom-left": (10, img.height - textheight - 10),
        "top-right": (img.width - textwidth - 10, 10),
        "top-left": (10, 10),
        "center": ((img.width - textwidth) // 2, (img.height - textheight) // 2)
    }
    pos = positions.get(position, positions["bottom-right"])
    draw.text(pos, text, fill=(255,255,255,int(255*float(opacity))), font=font)
    watermarked = Image.alpha_composite(img, watermark)
    buf = BytesIO()
    watermarked.save(buf, format="PNG")
    return buf.getvalue()

def save_uploaded_image_to_processed(file):
    """Save uploaded image to the processed folder with a unique name."""
    if file and allowed_file(file.filename):
        ext = os.path.splitext(secure_filename(file.filename))[1]
        unique_name = f"{uuid.uuid4().hex}{ext}"
        save_path = os.path.join(PROCESSED_FOLDER, unique_name)
        file.seek(0)  # Ensure pointer is at start
        file.save(save_path)
        file.seek(0)  # Reset pointer for further processing
        return save_path
    return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/image-tools')
def image_tools():
    return render_template('image_tools.html')

@app.route('/pdf-tools')
def pdf_tools():
    return render_template('pdf_tools.html')

@app.route('/compressor')
def compressor():
    return render_template('compressor.html')

@app.route('/resizer')
def resizer():
    return render_template('resizer.html')

@app.route('/converter')
def converter():
    return render_template('converter.html')

@app.route('/rotateflip')
def rotateflip():
    return render_template('rotateflip.html')

@app.route('/filters')
def filters():
    return render_template('filters.html')

@app.route('/sitemap.xml')
def sitemap():
    return send_from_directory('.', 'sitemap.xml')

@app.route('/adjust')
def adjust():
    return render_template('adjust.html')

@app.route('/adjust', methods=['POST'])
def adjust_api():
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image provided'}), 400
        file = request.files['image']
        brightness = float(request.form.get('brightness', 1.0))
        contrast = float(request.form.get('contrast', 1.0))
        color = float(request.form.get('color', 1.0))
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'Invalid file type'}), 400
        img_bytes = file.read()
        output_data = adjust_image(img_bytes, brightness, contrast, color)
        base64_image = base64.b64encode(output_data).decode('utf-8')
        return jsonify({'success': True, 'processed_image': f'data:image/png;base64,{base64_image}'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/crop')
def crop():
    return render_template('crop.html')

@app.route('/crop', methods=['POST'])
def crop_api():
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image provided'}), 400
        file = request.files['image']
        x = int(request.form.get('x', 0))
        y = int(request.form.get('y', 0))
        width = int(request.form.get('width', 100))
        height = int(request.form.get('height', 100))
        img_bytes = file.read()
        output_data = crop_image(img_bytes, x, y, width, height)
        base64_image = base64.b64encode(output_data).decode('utf-8')
        return jsonify({'success': True, 'processed_image': f'data:image/png;base64,{base64_image}'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/border')
def border():
    return render_template('border.html')

@app.route('/border', methods=['POST'])
def border_api():
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image provided'}), 400
        file = request.files['image']
        color = request.form.get('color', '#ffffff')
        thickness = int(request.form.get('thickness', 10))
        img_bytes = file.read()
        output_data = add_border(img_bytes, color, thickness)
        base64_image = base64.b64encode(output_data).decode('utf-8')
        return jsonify({'success': True, 'processed_image': f'data:image/png;base64,{base64_image}'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/rounded')
def rounded():
    return render_template('rounded.html')

@app.route('/rounded', methods=['POST'])
def rounded_api():
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image provided'}), 400
        file = request.files['image']
        radius = int(request.form.get('radius', 30))
        img_bytes = file.read()
        output_data = rounded_corners(img_bytes, radius)
        base64_image = base64.b64encode(output_data).decode('utf-8')
        return jsonify({'success': True, 'processed_image': f'data:image/png;base64,{base64_image}'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/watermark')
def watermark():
    return render_template('watermark.html')

@app.route('/watermark', methods=['POST'])
def watermark_api():
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image provided'}), 400
        file = request.files['image']
        text = request.form.get('text', 'Watermark')
        position = request.form.get('position', 'bottom-right')
        opacity = float(request.form.get('opacity', 0.5))
        img_bytes = file.read()
        output_data = add_watermark(img_bytes, text, position, opacity)
        base64_image = base64.b64encode(output_data).decode('utf-8')
        return jsonify({'success': True, 'processed_image': f'data:image/png;base64,{base64_image}'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/pdfmaker')
def pdfmaker():
    return render_template('pdfmaker.html')

@app.route('/pdfmaker', methods=['POST'])
def pdfmaker_api():
    # handle PDF creation here
    return jsonify({'success': False, 'error': 'Not implemented'}), 501

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
        
        # Save uploaded image to processed folder
        save_uploaded_image_to_processed(file)

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

@app.route('/compress', methods=['POST'])
def compress():
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image provided'}), 400
        file = request.files['image']
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'Invalid file type'}), 400
        quality = int(request.form.get('quality', 70))
        img_bytes = file.read()
        output_data = compress_image(img_bytes, quality)
        base64_image = base64.b64encode(output_data).decode('utf-8')
        return jsonify({'success': True, 'processed_image': f'data:image/jpeg;base64,{base64_image}'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/resize', methods=['POST'])
def resize():
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image provided'}), 400
        file = request.files['image']
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'Invalid file type'}), 400
        width = int(request.form.get('width', 256))
        height = int(request.form.get('height', 256))
        img_bytes = file.read()
        output_data = resize_image(img_bytes, width, height)
        base64_image = base64.b64encode(output_data).decode('utf-8')
        return jsonify({'success': True, 'processed_image': f'data:image/png;base64,{base64_image}'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/convert', methods=['POST'])
def convert():
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image provided'}), 400
        file = request.files['image']
        fmt = request.form.get('format', 'PNG').upper()
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'Invalid file type'}), 400
        img_bytes = file.read()
        output_data = convert_image(img_bytes, fmt)
        base64_image = base64.b64encode(output_data).decode('utf-8')
        return jsonify({'success': True, 'processed_image': f'data:image/{fmt.lower()};base64,{base64_image}'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/rotate', methods=['POST'])
def rotate():
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image provided'}), 400
        file = request.files['image']
        angle = float(request.form.get('angle', 90))
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'Invalid file type'}), 400
        img_bytes = file.read()
        output_data = rotate_image(img_bytes, angle)
        base64_image = base64.b64encode(output_data).decode('utf-8')
        return jsonify({'success': True, 'processed_image': f'data:image/png;base64,{base64_image}'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/flip', methods=['POST'])
def flip():
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image provided'}), 400
        file = request.files['image']
        mode = request.form.get('mode', 'horizontal')
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'Invalid file type'}), 400
        img_bytes = file.read()
        output_data = flip_image(img_bytes, mode)
        base64_image = base64.b64encode(output_data).decode('utf-8')
        return jsonify({'success': True, 'processed_image': f'data:image/png;base64,{base64_image}'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/filter', methods=['POST'])
def filter_img():
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image provided'}), 400
        file = request.files['image']
        filter_type = request.form.get('filter', 'grayscale')
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'Invalid file type'}), 400
        img_bytes = file.read()
        output_data = apply_filter(img_bytes, filter_type)
        base64_image = base64.b64encode(output_data).decode('utf-8')
        return jsonify({'success': True, 'processed_image': f'data:image/png;base64,{base64_image}'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/pdf', methods=['POST'])
def create_pdf():
    try:
        if 'images' not in request.files:
            return jsonify({'success': False, 'error': 'No images provided'}), 400
        files = request.files.getlist('images')
        if not files or any(f.filename == '' or not allowed_file(f.filename) for f in files):
            return jsonify({'success': False, 'error': 'Invalid file(s)'}), 400

        images = []
        for file in files:
            img = Image.open(file).convert("RGB")
            images.append(img)

        if not images:
            return jsonify({'success': False, 'error': 'No valid images'}), 400

        pdf_filename = f"{uuid.uuid4().hex}.pdf"
        pdf_path = os.path.join(PROCESSED_FOLDER, pdf_filename)
        images[0].save(pdf_path, save_all=True, append_images=images[1:], format='PDF')

        pdf_url = f"/download/{pdf_filename}"
        return jsonify({'success': True, 'pdf_url': pdf_url})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/download/<filename>')
def download_pdf(filename):
    return send_from_directory(PROCESSED_FOLDER, filename, as_attachment=True, mimetype='application/pdf')

@app.route('/pdf-split', methods=['POST'])
def pdf_split():
    try:
        if 'pdf' not in request.files:
            return jsonify({'success': False, 'error': 'No PDF provided'}), 400
        file = request.files['pdf']
        if file.filename == '' or not allowed_pdf(file.filename):
            return jsonify({'success': False, 'error': 'Invalid file type'}), 400
        pages = request.form.get('pages', '').replace(' ', '')
        if not pages:
            return jsonify({'success': False, 'error': 'No pages specified'}), 400

        reader = PdfReader(file)
        writer = PdfWriter()
        total_pages = len(reader.pages)
        page_ranges = []
        for part in pages.split(','):
            if '-' in part:
                start, end = part.split('-')
                page_ranges.extend(range(int(start)-1, int(end)))
            else:
                page_ranges.append(int(part)-1)
        for i in page_ranges:
            if 0 <= i < total_pages:
                writer.add_page(reader.pages[i])
        if not writer.pages:
            return jsonify({'success': False, 'error': 'No valid pages selected'}), 400

        out_name = f"{uuid.uuid4().hex}_split.pdf"
        out_path = os.path.join(PROCESSED_FOLDER, out_name)
        with open(out_path, "wb") as f:
            writer.write(f)
        return jsonify({'success': True, 'pdf_url': f"/download/{out_name}"})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/pdf-merge', methods=['POST'])
def pdf_merge():
    try:
        files = request.files.getlist('pdfs')
        if not files or any(f.filename == '' or not allowed_pdf(f.filename) for f in files):
            return jsonify({'success': False, 'error': 'Invalid file(s)'}), 400
        writer = PdfWriter()
        for file in files:
            reader = PdfReader(file)
            for page in reader.pages:
                writer.add_page(page)
        out_name = f"{uuid.uuid4().hex}_merged.pdf"
        out_path = os.path.join(PROCESSED_FOLDER, out_name)
        with open(out_path, "wb") as f:
            writer.write(f)
        return jsonify({'success': True, 'pdf_url': f"/download/{out_name}"})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/pdf-to-images', methods=['POST'])
def pdf_to_images():
    try:
        if 'pdf' not in request.files:
            return jsonify({'success': False, 'error': 'No PDF provided'}), 400
        file = request.files['pdf']
        if file.filename == '' or not allowed_pdf(file.filename):
            return jsonify({'success': False, 'error': 'Invalid file type'}), 400
        images = convert_from_bytes(file.read())
        urls = []
        for i, img in enumerate(images):
            img_name = f"{uuid.uuid4().hex}_page{i+1}.png"
            img_path = os.path.join(PROCESSED_FOLDER, img_name)
            img.save(img_path, "PNG")
            urls.append(f"/download/{img_name}")
        return jsonify({'success': True, 'images': urls})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/pdf-to-text', methods=['POST'])
def pdf_to_text():
    try:
        if 'pdf' not in request.files:
            return jsonify({'success': False, 'error': 'No PDF provided'}), 400
        file = request.files['pdf']
        if file.filename == '' or not allowed_pdf(file.filename):
            return jsonify({'success': False, 'error': 'Invalid file type'}), 400
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return jsonify({'success': True, 'text': text})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/pdf-compress', methods=['POST'])
def pdf_compress():
    try:
        if 'pdf' not in request.files:
            return jsonify({'success': False, 'error': 'No PDF provided'}), 400
        file = request.files['pdf']
        if file.filename == '' or not allowed_pdf(file.filename):
            return jsonify({'success': False, 'error': 'Invalid file type'}), 400

        target_size_kb = int(request.form.get('target_size', 500))
        reader = PdfReader(file)
        writer = PdfWriter()

        # Try to recompress images inside PDF pages
        for page in reader.pages:
            # Recompress images if present
            if '/XObject' in page.get('/Resources', {}):
                xObject = page['/Resources']['/XObject']
                for obj in xObject:
                    xobj = xObject[obj]
                    if xobj['/Subtype'] == '/Image':
                        try:
                            data = xobj.get_data()
                            img = Image.open(BytesIO(data))
                            buf = BytesIO()
                            # Reduce quality and size
                            img = img.convert("RGB")
                            img.save(buf, format="JPEG", quality=40, optimize=True)
                            buf.seek(0)
                            new_data = buf.read()
                            xobj._data = new_data
                            xobj[NameObject('/Filter')] = NameObject('/DCTDecode')
                            xobj[NameObject('/ColorSpace')] = NameObject('/DeviceRGB')
                            xobj[NameObject('/BitsPerComponent')] = 8
                            xobj[NameObject('/Length')] = len(new_data)
                        except Exception:
                            pass  # If image can't be recompressed, skip
            writer.add_page(page)

        compressed_filename = f"{uuid.uuid4().hex}_compressed.pdf"
        compressed_path = os.path.join(PROCESSED_FOLDER, compressed_filename)
        with open(compressed_path, "wb") as f:
            writer.write(f)

        # Try to trim further if needed by removing metadata
        actual_size_kb = os.path.getsize(compressed_path) // 1024
        msg = None
        if actual_size_kb > target_size_kb:
            msg = f"Compressed PDF is {actual_size_kb} KB, larger than target {target_size_kb} KB. Pure Python compression is limited; for better results, use external tools like Ghostscript."
        return jsonify({'success': True, 'pdf_url': f"/download/{compressed_filename}", 'note': msg})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/pdf-to-word', methods=['POST'])
def pdf_to_word():
    try:
        if 'pdf' not in request.files:
            return jsonify({'success': False, 'error': 'No PDF provided'}), 400
        file = request.files['pdf']
        if file.filename == '' or not allowed_pdf(file.filename):
            return jsonify({'success': False, 'error': 'Invalid file type'}), 400

        pdf_filename = f"{uuid.uuid4().hex}.pdf"
        pdf_path = os.path.join(PROCESSED_FOLDER, pdf_filename)
        file.save(pdf_path)

        docx_filename = f"{uuid.uuid4().hex}.docx"
        docx_path = os.path.join(PROCESSED_FOLDER, docx_filename)

        cv = Converter(pdf_path)
        cv.convert(docx_path, start=0, end=None)
        cv.close()

        return jsonify({'success': True, 'word_url': f"/download/{docx_filename}"})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/pdf-to-excel', methods=['POST'])
def pdf_to_excel():
    try:
        if 'pdf' not in request.files:
            return jsonify({'success': False, 'error': 'No PDF provided'}), 400
        file = request.files['pdf']
        if file.filename == '' or not allowed_pdf(file.filename):
            return jsonify({'success': False, 'error': 'Invalid file type'}), 400

        pdf_filename = f"{uuid.uuid4().hex}.pdf"
        pdf_path = os.path.join(PROCESSED_FOLDER, pdf_filename)
        file.save(pdf_path)

        excel_filename = f"{uuid.uuid4().hex}.xlsx"
        excel_path = os.path.join(PROCESSED_FOLDER, excel_filename)

        wb = openpyxl.Workbook()
        ws = wb.active

        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for table in tables:
                    for row in table:
                        ws.append(row)
        wb.save(excel_path)

        return jsonify({'success': True, 'excel_url': f"/download/{excel_filename}"})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/pdf-to-html', methods=['POST'])
def pdf_to_html():
    try:
        if 'pdf' not in request.files:
            return jsonify({'success': False, 'error': 'No PDF provided'}), 400
        file = request.files['pdf']
        if file.filename == '' or not allowed_pdf(file.filename):
            return jsonify({'success': False, 'error': 'Invalid file type'}), 400

        pdf_filename = f"{uuid.uuid4().hex}.pdf"
        pdf_path = os.path.join(PROCESSED_FOLDER, pdf_filename)
        file.save(pdf_path)

        html_filename = f"{uuid.uuid4().hex}.html"
        html_path = os.path.join(PROCESSED_FOLDER, html_filename)

        with open(pdf_path, "rb") as fin, open(html_path, "w", encoding="utf-8") as fout:
            extract_text_to_fp(fin, fout, output_type='html', laparams=None)

        return jsonify({'success': True, 'html_url': f"/download/{html_filename}"})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
    app.run(host="127.0.0.1", port=5000)