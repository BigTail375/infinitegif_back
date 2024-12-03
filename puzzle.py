from PIL import Image

def overlay_images(jpg_path, png_path, output_path):
    # Load images
    jpg_image = Image.open(jpg_path)
    png_image = Image.open(png_path)

    # Get dimensions
    jpg_width, jpg_height = jpg_image.size
    png_width, png_height = png_image.size

    # Determine aspect ratios
    jpg_aspect = jpg_width / jpg_height
    png_aspect = png_width / png_height

    # Rotate PNG if necessary
    if (png_width < png_height and jpg_width > jpg_height) or (png_width > png_height and jpg_width < jpg_height):
        png_image = png_image.rotate(90, expand=True)
        png_width, png_height = png_image.size
        png_aspect = png_width / png_height

    # Crop PNG to match JPG aspect ratio
    if png_aspect > jpg_aspect:
        # PNG is wider than JPG
        new_width = int(png_height * jpg_aspect)
        left = (png_width - new_width) / 2
        right = left + new_width
        top = 0
        bottom = png_height
    else:
        # PNG is taller than JPG
        new_height = int(png_width / jpg_aspect)
        top = (png_height - new_height) / 2
        bottom = top + new_height
        left = 0
        right = png_width

    png_image = png_image.crop((left, top, right, bottom))

    # Resize PNG to match JPG size
    png_image = png_image.resize((jpg_width, jpg_height), Image.LANCZOS)

    # Overlay PNG on JPG
    jpg_image.paste(png_image, (0, 0), png_image)

    # Save the result
    jpg_image.save(output_path)