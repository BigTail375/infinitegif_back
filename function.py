from PIL import Image
import io

def convert_image_to_bytesio(file_path):
    # Open the image file
    with Image.open(file_path) as img:
        # Create a BytesIO object
        img_byte_arr = io.BytesIO()
        # Save the image to the BytesIO object in PNG format
        img.save(img_byte_arr, format='PNG')
        # Seek to the start of the BytesIO object
        img_byte_arr.seek(0)
    return img_byte_arr