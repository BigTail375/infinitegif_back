import os
import time
import shutil

from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
from pymongo import MongoClient, DESCENDING
from flask_cors import CORS
import uuid
from pydantic import BaseModel
from typing import List
from PIL import Image, ImageSequence
from dotenv import load_dotenv
import io

from zoom import image2recrusive
from path import IMG_DIR, TEMP_DIR

# MongoDB connection
client = MongoClient("mongodb://localhost:27017")
db = client["image_database"]
collection = db["images"]

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Pydantic model for response validation
class ImageResponse(BaseModel):
    url: str
    tags: List[str]

@app.route('/page', methods=['POST'])
def get_page():
    try:
        img_per_page = 10

        # Get JSON data from the request
        data = request.get_json()
 
        # Extract 'number' from the request data
        page_index = data.get('page')

        # Check if 'number' is an integer
        if page_index is None:
            return jsonify({'error': 'No number provided'}), 400
        if not isinstance(page_index, int):
            return jsonify({'error': 'The number must be an integer'}), 400
        
        img_count = len(os.listdir(IMG_DIR)) - 1
        data = []
        end_index = img_count - img_per_page * (page_index - 1)
        start_index = end_index - img_per_page
        if start_index < 0:
            start_index = 0
        for i in range(img_count - 1, start_index - 1, -1):
            if os.path.exists(os.path.join(IMG_DIR, f'{i}.gif')):
                data.append({'path': f'{i}.gif'})
            else:
                data.append({'path': f'{i}.png'})
        
        # Return the number as part of the response
        return jsonify({'results': data}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if the POST request has the file part
    if 'file' not in request.files:
        return {'error': 'No file part'}, 400
    
    file = request.files['file']

    if file.filename == '':
        return {'error': 'No selected file'}, 400

    if file:
        img_count = len(os.listdir(IMG_DIR)) - 1
        file_path = os.path.join(IMG_DIR, f'{img_count}.gif')
        file.save(file_path)

        gif = Image.open(file_path)
        watermark = Image.open('watermark.png')
        
        new_width = int(gif.size[0] * 0.2)
        new_height = int(new_width / watermark.size[0] * watermark.size[1])
        watermark_resized = watermark.resize((new_width, new_height), Image.LANCZOS)

        position = (int(gif.size[0] * 0.7), int(gif.size[1] * 0.1))

        frames = []

        for frame in ImageSequence.Iterator(gif):
            frame = frame.convert('RGBA')
            frame_with_watermark = frame.copy()
            frame_with_watermark.paste(watermark_resized, position, watermark_resized)
            frames.append(frame_with_watermark)
        frames[0].save(file_path, save_all=True, append_images=frames[1:], loop=0, duration=gif.info['duration'])

        return {'message': 'File saved successfully'}, 200

@app.route('/gif2grid', methods=['POST'])
def gif_grid():
    # Check if the POST request has the file part
    if 'file' not in request.files:
        return {'error': 'No file part'}, 400
    
    file = request.files['file']
    grid_size = request.form['gridSize']

    if file.filename == '':
        return {'error': 'No selected file'}, 400

    if file:
        try:
            rows, cols = map(int, grid_size.split('x'))  # Extract rows and columns from grid size
        except ValueError:
            return {'error': 'Invalid grid size format'}, 400
        
        file_path = os.path.join(TEMP_DIR, f'{time.time()}.gif')
        file.save(file_path)

        gif = Image.open(file_path)

        # Calculate frame step to sample evenly from the GIF
        frame_count = gif.n_frames
        total_images = rows * cols

        # Create a new blank image for the grid
        gif_width, gif_height = gif.size
        grid_image = Image.new('RGBA', (cols * gif_width, rows * gif_height))
        
        frames = []
        for i in range(total_images):
            gif.seek(int(frame_count * i / total_images))
            frames.append(gif.copy())

        for i, frame in enumerate(frames):
            row = i // cols
            col = i % cols
            grid_image.paste(frame, (col * gif_width, row * gif_height))

        # Save the grid image
        grid_img_name = f'{time.time()}.png'
        grid_img_path = os.path.join(TEMP_DIR, grid_img_name)
        grid_image.save(grid_img_path)
        return jsonify({'results': grid_img_name}), 200

    return {'error': 'File processing error'}, 500

@app.route('/uploadGrid', methods=['POST'])
def uploadGrid():
    try:
        file_path = request.form['file']
        
        img_count = len(os.listdir(IMG_DIR)) - 1
        shutil.copy(os.path.join(TEMP_DIR, file_path), os.path.join(IMG_DIR, f'{img_count}.png'))
        
        return jsonify({'success': 'image is uploaded!'}), 200

    except Exception as e:
        print (e)
        return jsonify({'error': str(e)}), 500

@app.route('/recrusivegif', methods=['POST'])
def recrusiveGif():
    try:
        if 'file' not in request.files:
            return "No file part", 400
        file = request.files['file']
        if file.filename == '':
            return "No selected file", 400

        # Process the image (example: convert to grayscale)
        img = Image.open(file.stream)
        result_img = image2recrusive(img)

        # Save processed image to a byte stream
        img_byte_arr = io.BytesIO()
        result_img[0].save(
            img_byte_arr,
            format='GIF',
            save_all=True,
            append_images=result_img[1:],
            loop=0,
            duration=100  # Adjust duration per frame if needed
        )
        img_byte_arr.seek(0)

        return send_file(img_byte_arr, mimetype='image/png')
    except Exception as e:
        print (e)
        return jsonify({'error', str(e)}), 500


if __name__ == '__main__':
    app.run()