import numpy as np
from PIL import Image
import imageio
import random

def create_puzzle_effect(image_path, output_gif, num_pieces=4, duration=0.1):
    try:
        # Load the image
        image = Image.open(image_path)
        width, height = image.size

        # Calculate piece size
        piece_width = width // num_pieces
        piece_height = height // num_pieces

        # Create a list to hold the pieces
        pieces = []

        # Cut the image into pieces
        for i in range(num_pieces):
            for j in range(num_pieces):
                box = (j * piece_width, i * piece_height, (j + 1) * piece_width, (i + 1) * piece_height)
                piece = image.crop(box)
                pieces.append(piece)

        # Shuffle pieces for the initial scrambled state
        random.shuffle(pieces)

        # Create frames for the GIF
        frames = []

        # Create a blank image for assembling the puzzle
        blank_image = Image.new('RGB', (width, height), (255, 255, 255))

        # Animate the pieces moving into place
        for step in range(num_pieces * num_pieces):
            frame = blank_image.copy()
            for index, piece in enumerate(pieces):
                if index <= step:
                    x = (index % num_pieces) * piece_width
                    y = (index // num_pieces) * piece_height
                    frame.paste(piece, (x, y))
            frames.append(frame)

        # Save frames as a GIF
        imageio.mimsave(output_gif, frames, format='GIF', duration=duration)
        print(f"GIF saved successfully as {output_gif}")

    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
create_puzzle_effect('01-happy-asian-freelancer-developer-man-at-office_slidesbase-1.jpg', 'puzzle_effect.gif')