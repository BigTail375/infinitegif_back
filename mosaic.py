from PIL import Image
import numpy as np

def apply_mosaic_effect(image_path, output_path, tile_size=10):
    # Open the image
    image = Image.open(image_path)
    image = image.convert('RGB')  # Ensure image is in RGB format
    width, height = image.size

    # Convert image to numpy array
    image_array = np.array(image)

    # Process each tile
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            # Define the region of the tile
            x_end = min(x + tile_size, width)
            y_end = min(y + tile_size, height)

            # Extract the tile
            tile = image_array[y:y_end, x:x_end]

            # Calculate the average color of the tile
            average_color = tile.mean(axis=(0, 1)).astype(int)

            # Fill the tile with the average color
            image_array[y:y_end, x:x_end] = average_color

    # Convert the array back to an image
    mosaic_image = Image.fromarray(image_array)

    # Save the mosaic image
    mosaic_image.save(output_path)
    print(f"Mosaic effect applied and saved as {output_path}")

# Example usage
apply_mosaic_effect('01-happy-asian-freelancer-developer-man-at-office_slidesbase-1.jpg', 'mosaic_effect.jpg', tile_size=5)