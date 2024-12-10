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

def resize_and_save_image(file_path, output_path=None):
    # Open the image file
    with Image.open(file_path) as img:
        # Get original dimensions
        original_width, original_height = img.size
        
        # Determine the scaling factor
        if original_width > 900 or original_height > 900:
            if original_width > original_height:
                scaling_factor = 900 / original_width
            else:
                scaling_factor = 900 / original_height
            
            # Calculate new dimensions
            new_width = int(original_width * scaling_factor)
            new_height = int(original_height * scaling_factor)
            
            # Resize the image
            img = img.resize((new_width, new_height), Image.LANCZOS)
        
        # Determine the output path
        if output_path is None:
            output_path = file_path  # Overwrite the original file
        
        # Save the resized image
        if file_path.endswith('.png'):
            img.save(output_path, format='PNG')
        else:
            img.save(output_path, format='JPEG')