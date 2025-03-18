import os
import time
import shutil
import cv2

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
from datetime import datetime
import json
from bson import ObjectId
import requests

from path import IMG_DIR, TEMP_DIR, ML_SERVER
from utils.zoom import image2recrusive
from utils.paintbynumber import paint_by_number
from utils.puzzle import overlay_images
from utils.function import convert_image_to_bytesio
from mosaic.mosaic import create_roman_mosaic
from utils.effect import apply_effect
# MongoDB connection
client = MongoClient("mongodb://localhost:27017")
db = client["image_database"]
collection_image = db["images"]
collection_audio = db["audio"]

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Pydantic model for response validation
class ImageResponse(BaseModel):
    url: str
    tags: List[str]

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return {'error': 'No file part'}, 400
    
    file = request.files['file']
    tags = request.form.get('tags', '[]')

    if file.filename == '':
        return {'error': 'No selected file'}, 400
    try:
        tags = json.loads(tags)  # Parse the tags
    except json.JSONDecodeError:
        return {'error': 'Invalid tags format'}, 400
    
    print (tags, type(tags))
    if len(tags) > 0 and tags[0] == "":
        tags = []
    file_ext = file.filename.split('.')[-1]
    if file_ext == 'extension':
        file_ext = 'gif'
    filename = f"{uuid.uuid4()}.{file_ext}"
    file_path = os.path.join(IMG_DIR, secure_filename(filename))
    file.save(file_path)

    document = {
        "folder_path": filename,
        "tags": tags,
        "audio_path": "",
        "vote_count": 0,
        "upload_time": datetime.now().timestamp() # Unique value for sorting by upload time
    }
    collection_image.insert_one(document)

    return jsonify({"message": "Image uploaded successfully", "file_path": file_path})

@app.route('/page', methods=['POST'])
def get_page():
    try:
        img_per_page = 10
        data = request.get_json()
        page_index = data.get('page')
        skip_index = data.get('skip')

        document_count = collection_image.count_documents({}) - skip_index
        img_count = img_per_page * (page_index + 1)
        if img_count > document_count:
            img_count = document_count
        cursor = collection_image.find().sort('upload_time', DESCENDING).skip(skip_index).limit(img_count)
        images = list(cursor)

        if not images:
            return jsonify({"error": "No images found"}), 404
        response = [{"path": image["folder_path"], "_id": str(image["_id"]), "audio": image["audio_path"]} for image in images]
        return jsonify({'results': response}), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/audio', methods=['POST'])
def get_audio():
    try:
        audio_per_page = 10
        data = request.get_json()
        page_index = data.get('page')
        skip_index = data.get('skip')

        document_count = collection_audio.count_documents({}) - skip_index
        audio_count = audio_per_page * (page_index + 1)
        if audio_count > document_count:
            audio_count = document_count
        cursor = collection_audio.find().sort('upload_time', DESCENDING).skip(skip_index).limit(audio_count)
        audios = list(cursor)

        if not audios:
            return jsonify({"error": "No images found"}), 404
        response = [{"_id": str(audio["_id"]), "audio": audio["folder_path"]} for audio in audios]
        return jsonify({'results': response}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route("/image_by_tags", methods=["POST"])
def get_images_by_tags():
    # try:
    image_per_page = 10

    data = request.json
    page = data.get("page")
    tags = data.get("tags", [])
    skip_index = data.get("skip")
    
    filter_query = {"tags": {"$in": tags}}

    img_count = image_per_page * (int(page) + 1)
    tag_collection_count = collection_image.count_documents(filter_query) - skip_index
    print (tag_collection_count)
    if img_count > tag_collection_count:
        img_count = tag_collection_count

    cursor = collection_image.find(filter_query).sort('upload_time', DESCENDING).skip(skip_index).limit(img_count)
    images = list(cursor)

    if not images:
        return jsonify({"error": "No images found"}), 404
    response = [{"path": image["folder_path"], "_id": str(image["_id"]), "audio": image["audio_path"]} for image in images]
    return jsonify({'results': response}), 200
    # except Exception as e:
    #     return jsonify({'error': str(e)}), 500

@app.route('/id', methods=['POST'])
def get_id():
    try:
        data = request.get_json()
        _id = data.get('id')
        object_id = ObjectId(str(_id))
        image = collection_image.find_one({"_id": object_id})

        if not image:
            return jsonify({"error": "No images found"}), 404
        response = {'path': image['folder_path'], 'tags': image['tags'], "audio": image["audio_path"], "vote": image["vote_count"]}
        
        return jsonify({'results': response}), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/update_vote', methods=['POST'])
def update_vote():
    try:
        data = request.get_json()
        _id = data.get('id')
        vote = data.get('vote')
        object_id = ObjectId(str(_id))
        image = collection_image.find_one({"_id": object_id})

        if vote == "up":
            collection_image.update_one(
                {"_id": object_id},
                {"$inc": {"vote_count": 1}}
            )
        else:
            collection_image.update_one(
                {"_id": object_id},
                {"$inc": {"vote_count": -1}}
            )
        return jsonify({'results': "vote is updated!"}), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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
        grid_img_name = f"{uuid.uuid4()}.png"
        grid_img_path = os.path.join(TEMP_DIR, grid_img_name)
        grid_image.save(grid_img_path)
        return jsonify({'results': grid_img_name}), 200

    return {'error': 'File processing error'}, 500

@app.route('/uploadGrid', methods=['POST'])
def uploadGrid():
    try:
        file_path = request.form['file']
        
        shutil.move(os.path.join(TEMP_DIR, file_path), os.path.join(IMG_DIR, file_path))
        tags = []
        document = {
            "folder_path": file_path,
            "tags": tags,
            "audio_path": "",
            "vote_count": 0,
            "upload_time": datetime.now().timestamp() # Unique value for sorting by upload time
        }
        collection_image.insert_one(document)

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

@app.route('/paintbynumber', methods=['POST'])
def paintbynumber():
    try:
        if 'file' not in request.files:
            return "No file part", 400
        file = request.files['file']
        if file.filename == '':
            return "No selected file", 400

        file_path = os.path.join(TEMP_DIR, f'{time.time()}.jpg')
        file.save(file_path)
        painted_img = paint_by_number(file_path)
        _, buffer = cv2.imencode('.jpg', painted_img)
        img_byte_arr = io.BytesIO(buffer.tobytes())
        img_byte_arr.seek(0)

        return send_file(img_byte_arr, mimetype='image/png')
    except Exception as e:
        print (e)
        return jsonify({'error', str(e)}), 500

@app.route('/puzzle', methods=['POST'])
def puzzle():
    try:
        if 'file' not in request.files:
            return "No file part", 400
        file = request.files['file']
        if file.filename == '':
            return "No selected file", 400
        
        piece_size = int(request.form['pieceSize'])
        if piece_size > 5:
            piece_size = 4
        file_path = os.path.join(TEMP_DIR, f'{time.time()}.jpg')
        saved_file_path = os.path.join(TEMP_DIR, f'{time.time()}.jpg')
        file.save(file_path)
        overlay_images(file_path, 'assets/overlay.png', saved_file_path)
        img_byte_arr = convert_image_to_bytesio(saved_file_path)
        return send_file(img_byte_arr, mimetype='image/png')
    except Exception as e:
        print (e)
        return jsonify({'error', str(e)}), 500

@app.route('/effect', methods=['POST'])
def effect():
    try:
        if 'file' not in request.files:
            return "No file part", 400
        file = request.files['file']
        if file.filename == '':
            return "No selected file", 400
        
        effect_type = int(request.form['effectType'])
        param1, param2, param3 = None, None, None
        if effect_type in [0, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19]:
            param1 = float(request.form['param1'])
        if effect_type in [0, 14]:
            param2 = float(request.form['param2'])
        if effect_type == 14:
            param3 = float(request.form['param3'])
        file_path = os.path.join(TEMP_DIR, f'{time.time()}.jpg')
        saved_file_path = os.path.join(TEMP_DIR, f'{time.time()}.jpg')
        file.save(file_path)
        apply_effect(file_path, saved_file_path, effect_type, param1, param2, param3)
        img_byte_arr = convert_image_to_bytesio(saved_file_path)
        return send_file(img_byte_arr, mimetype='image/png')
    except Exception as e:
        print (e)
        return jsonify({'error', str(e)}), 500

@app.route('/mosaic', methods=['POST'])
def mosaic():
    try:
        if 'file' not in request.files:
            return "No file part", 400
        file = request.files['file']
        if file.filename == '':
            return "No selected file", 400
        
        tile_size = int(request.form['tileSize'])
        file_path = os.path.join(TEMP_DIR, f'{time.time()}.jpg')
        saved_file_path = os.path.join(TEMP_DIR, f'{time.time()}.jpg')
        file.save(file_path)
        rotated_file_path = os.path.join(TEMP_DIR, f'{time.time()}.jpg')
        rotate_flag = False
        with Image.open(file_path) as img:
            if img.width < img.height:
                img = img.rotate(-90, expand=True)
                img.save(rotated_file_path)
                rotate_flag = True
        if rotate_flag:
            create_roman_mosaic(rotated_file_path, saved_file_path)
            with Image.open(saved_file_path) as img:
                img = img.rotate(90, expand=True)
                img.save(rotated_file_path)
            img_byte_arr = convert_image_to_bytesio(rotated_file_path)
        else:
            create_roman_mosaic(file_path, saved_file_path)
            img_byte_arr = convert_image_to_bytesio(saved_file_path)

        return send_file(img_byte_arr, mimetype='image/png')
    except Exception as e:
        print (e)
        return jsonify({'error', str(e)}), 500

@app.route('/url2grid', methods=['POST'])
def gif_grid_url():
    _id = request.form['_id']
    grid_size = request.form['gridSize']
    
    try:
        rows, cols = map(int, grid_size.split('x'))  # Extract rows and columns from grid size
    except ValueError:
        return {'error': 'Invalid grid size format'}, 400
    
    
    object_id = ObjectId(str(_id))
    image = collection_image.find_one({"_id": object_id})

    file_path = os.path.join(IMG_DIR, image['folder_path'])

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
    grid_img_name = f"{uuid.uuid4()}.png"
    grid_img_path = os.path.join(TEMP_DIR, grid_img_name)
    grid_image.save(grid_img_path)
    return jsonify({'results': grid_img_name}), 200
 
@app.route('/url2recrusive', methods=['POST'])
def url2recrusive():
    try:
        _id = request.form['_id']
        object_id = ObjectId(str(_id))

        image = collection_image.find_one({"_id": object_id})
        file_path = os.path.join(IMG_DIR, image['folder_path'])
        img = Image.open(file_path)
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

@app.route('/url2paint', methods=['POST'])
def url2paint():
    try:
        _id = request.form['_id']
        object_id = ObjectId(str(_id))

        image = collection_image.find_one({"_id": object_id})
        file_path = os.path.join(IMG_DIR, image['folder_path'])

        painted_img = paint_by_number(file_path)
        _, buffer = cv2.imencode('.jpg', painted_img)
        img_byte_arr = io.BytesIO(buffer.tobytes())
        img_byte_arr.seek(0)

        return send_file(img_byte_arr, mimetype='image/png')
    except Exception as e:
        print (e)
        return jsonify({'error', str(e)}), 500

@app.route('/url2puzzle', methods=['POST'])
def url2puzzle():
    try:
        _id = request.form['_id']
        piece_size = int(request.form['pieceSize'])
        if piece_size > 5:
            piece_size = 4
        object_id = ObjectId(str(_id))

        image = collection_image.find_one({"_id": object_id})
        file_path = os.path.join(IMG_DIR, image['folder_path'])
        saved_file_path = os.path.join(TEMP_DIR, f'{time.time()}.png')
        overlay_images(file_path, 'assets/overlay.png', saved_file_path)
        img_byte_arr = convert_image_to_bytesio(saved_file_path)

        return send_file(img_byte_arr, mimetype='image/png')
    except Exception as e:
        print (e)
        return jsonify({'error', str(e)}), 500

@app.route('/url2mosaic', methods=['POST'])
def url2mosaic():
    try:
        _id = request.form['_id']
        tile_size = int(request.form['tileSize'])
        object_id = ObjectId(str(_id))

        image = collection_image.find_one({"_id": object_id})
        file_path = os.path.join(IMG_DIR, image['folder_path'])
        saved_file_path = os.path.join(TEMP_DIR, f'{time.time()}.jpg')
        rotated_file_path = os.path.join(TEMP_DIR, f'{time.time()}.jpg')
        rotate_flag = False
        with Image.open(file_path) as img:
            if img.width < img.height:
                img = img.rotate(-90, expand=True)
                img.save(rotated_file_path)
                rotate_flag = True
        if rotate_flag:
            create_roman_mosaic(rotated_file_path, saved_file_path)
            with Image.open(saved_file_path) as img:
                img = img.rotate(90, expand=True)
                img.save(rotated_file_path)
            img_byte_arr = convert_image_to_bytesio(rotated_file_path)
        else:
            create_roman_mosaic(file_path, saved_file_path)
            img_byte_arr = convert_image_to_bytesio(saved_file_path)

        return send_file(img_byte_arr, mimetype='image/png')
    except Exception as e:
        print (e)
        return jsonify({'error', str(e)}), 500

@app.route('/url2effect', methods=['POST'])
def url2effect():
    try:
        _id = request.form['_id']
        object_id = ObjectId(str(_id))

        image = collection_image.find_one({"_id": object_id})
        file_path = os.path.join(IMG_DIR, image['folder_path'])
        saved_file_path = os.path.join(TEMP_DIR, f'{time.time()}.png')
        
        effect_type = int(request.form['effectType'])
        param1, param2, param3 = None, None, None
        if effect_type in [0, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19]:
            param1 = float(request.form['param1'])
        if effect_type in [0, 14]:
            param2 = float(request.form['param2'])
        if effect_type == 14:
            param3 = float(request.form['param3'])
        apply_effect(file_path, saved_file_path, effect_type, param1, param2, param3)
        img_byte_arr = convert_image_to_bytesio(saved_file_path)
        return send_file(img_byte_arr, mimetype='image/png')
    except Exception as e:
        print (e)
        return jsonify({'error', str(e)}), 500

@app.route('/faceswap', methods=['POST'])
def faceswap():
    if 'file' not in request.files:
        return {'error': 'No file part'}, 400

    _id = request.form['_id']
    source_file = request.files['file']
    if source_file.filename == '':
            return {'error': 'No selected file'}, 400
    source_file_ext = source_file.filename.split('.')[-1]
    source_filename = f"{uuid.uuid4()}.{source_file_ext}"
    source_file_path = os.path.join(TEMP_DIR, secure_filename(source_filename))
    source_file.save(source_file_path)

    object_id = ObjectId(str(_id))
    target_image = collection_image.find_one({"_id": object_id})
    target_file_path = os.path.join(IMG_DIR, target_image['folder_path'])
    
    # Save the face swap image
    faceswap_img_name = f"{uuid.uuid4()}.png"
    faceswap_img_path = os.path.join(TEMP_DIR, faceswap_img_name)

    with open(target_file_path, 'rb') as target_file, open(source_file_path, 'rb') as source_file:
        # Prepare the files and form data
        files = {
            'target': target_file,
            'source': source_file
        }
        data = {
            'face_enhancer': 'False'  # or 'false' depending on your requirement
        }
        
        # Send the POST request
        response = requests.post(ML_SERVER, files=files, data=data)

        # Check if the request was successful
        if response.status_code == 200:
            # Save the received image file
            with open(faceswap_img_path, 'wb') as output_file:  # Change the extension if needed
                output_file.write(response.content)
            print("File saved successfully.")
        else:
            print(f"Request failed with status code: {response.status_code}")
            return {'error': 'Face Swap Failed!'}, 400

    return jsonify({'results': faceswap_img_name}), 200

if __name__ == '__main__':
    # app.run()
    app.run(port=5001)