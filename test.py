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
from datetime import datetime
import json
from bson import ObjectId

from path import IMG_DIR, TEMP_DIR
from zoom import image2recrusive


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

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return {'error': 'No file part'}, 400
    
    file = request.files['file']
    tags = request.form.getlist("tags")
    tags = json.loads(tags[0])

    if file.filename == '':
        return {'error': 'No selected file'}, 400
    
    print (tags, type(tags))

    filename = f"{uuid.uuid4()}.{file.filename.split('.')[-1]}"
    file_path = os.path.join(IMG_DIR, secure_filename(filename))
    file.save(file_path)

    document = {
        "folder_path": filename,
        "tags": tags,
        "upload_time": datetime.now().timestamp() # Unique value for sorting by upload time
    }
    collection.insert_one(document)

    return jsonify({"message": "Image uploaded successfully", "file_path": file_path})

@app.route('/page', methods=['POST'])
def get_page():
    try:
        img_per_page = 10
        data = request.get_json()
        page_index = data.get('page')

        document_count = collection.count_documents({})
        img_count = img_per_page * (page_index + 1)
        if img_count > document_count:
            img_count = document_count
        cursor = collection.find().sort('upload_time', DESCENDING).limit(img_count)
        images = list(cursor)

        if not images:
            return jsonify({"error": "No images found"}), 404
        response = [{"path": image["folder_path"], "_id": str(image["_id"])} for image in images]
        return jsonify({'results': response}), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/id', methods=['POST'])
def get_id():
    try:
        data = request.get_json()
        _id = data.get('id')
        object_id = ObjectId(str(_id))
        image = collection.find_one({"_id": object_id})

        if not image:
            return jsonify({"error": "No images found"}), 404
        response = {'path': image['folder_path'], 'tags': image['tags']}
        
        return jsonify({'results': response}), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route("/image_by_tags", methods=["POST"])
def get_images_by_tags():
    # try:
    image_per_page = 10

    page = request.form["page"]
    tags = request.form.getlist("tags")
    tags = json.loads(tags[0])
    filter_query = {"tags": {"$in": tags}}

    img_count = image_per_page * (int(page) + 1)
    tag_collection_count = collection.count_documents(filter_query)
    print (tag_collection_count)
    if img_count > tag_collection_count:
        img_count = tag_collection_count

    cursor = collection.find(filter_query).sort('upload_time', DESCENDING).limit(img_count)
    images = list(cursor)

    if not images:
        return jsonify({"error": "No images found"}), 404
    response = [{"path": image["folder_path"]} for image in images]
    return jsonify({'results': response}), 200
    # except Exception as e:
    #     return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # app.run()
    app.run(port=5001)