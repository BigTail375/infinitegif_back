import cv2
import cv2.xphoto
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps

def oil_painting(input_file_path, output_file_path, radius = 7, sigma = 1, param1 = None):
    img = cv2.imread(input_file_path)
    res = cv2.xphoto.oilPainting(img, radius, sigma)
    cv2.imwrite(output_file_path, res)
    return res

def cartoon(input_file_path, output_file_path, param1, param2, param3):
    image = cv2.imread(input_file_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(image, 9, 300, 300)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    cv2.imwrite(output_file_path, cartoon)
    return cartoon

def grayscale(input_file_path, output_file_path, param1, param2, param3):
    image = Image.open(input_file_path).convert("L")
    image.save(output_file_path)

def sepia(input_file_path, output_file_path, param1, param2, param3):
    image = Image.open(input_file_path)
    sepia_filter = ImageEnhance.Color(image).enhance(0.3)
    sepia_filter.save(output_file_path)

def negative(input_file_path, output_file_path, param1, param2, param3):
    image = Image.open(input_file_path)
    inverted_image = ImageOps.invert(image)
    inverted_image.save(output_file_path)

def blur(input_file_path, output_file_path, param1, param2, param3):
    image = Image.open(input_file_path)
    blurred_image = image.filter(ImageFilter.BLUR)
    blurred_image.save(output_file_path)

def sharpen(input_file_path, output_file_path, param1, param2, param3):
    image = Image.open(input_file_path)
    sharpened_image = image.filter(ImageFilter.SHARPEN)
    sharpened_image.save(output_file_path)

def edge_detection(input_file_path, output_file_path, param1, param2, param3):
    image = Image.open(input_file_path)
    edges = image.filter(ImageFilter.FIND_EDGES)
    edges.save(output_file_path)

def emboss(input_file_path, output_file_path, param1, param2, param3):
    image = Image.open(input_file_path)
    embossed_image = image.filter(ImageFilter.EMBOSS)
    embossed_image.save(output_file_path)

def brightness_adjustment(input_file_path, output_file_path, factor, param1, param2):
    image = Image.open(input_file_path)
    enhancer = ImageEnhance.Brightness(image)
    brightened_image = enhancer.enhance(factor)
    brightened_image.save(output_file_path)

def contrast_adjustment(input_file_path, output_file_path, factor, param1, param2):
    image = Image.open(input_file_path)
    enhancer = ImageEnhance.Contrast(image)
    contrasted_image = enhancer.enhance(factor)
    contrasted_image.save(output_file_path)

def saturation_adjustment(input_file_path, output_file_path, factor, param1, param2):
    image = Image.open(input_file_path)
    enhancer = ImageEnhance.Color(image)
    saturated_image = enhancer.enhance(factor)
    saturated_image.save(output_file_path)

def gaussian_blur(input_file_path, output_file_path, radius, param1, param2):
    image = Image.open(input_file_path)
    blurred_image = image.filter(ImageFilter.GaussianBlur(radius))
    blurred_image.save(output_file_path)

def median_filter(input_file_path, output_file_path, size, param1, param2):
    image = Image.open(input_file_path)
    filtered_image = image.filter(ImageFilter.MedianFilter(size))
    filtered_image.save(output_file_path)

def bilateral_filter(input_file_path, output_file_path, d, sigma_color, sigma_space):
    image = cv2.imread(input_file_path)
    filtered_image = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    cv2.imwrite(output_file_path, filtered_image)

def thresholding(input_file_path, output_file_path, threshold, param1, param2):
    image = Image.open(input_file_path).convert("L")
    thresholded_image = image.point(lambda p: p > threshold and 255)
    thresholded_image.save(output_file_path)

def posterize(input_file_path, output_file_path, bits, param1, param2):
    image = Image.open(input_file_path)
    posterized_image = ImageOps.posterize(image, bits)
    posterized_image.save(output_file_path)

def solarize(input_file_path, output_file_path, threshold, param1, param2):
    image = Image.open(input_file_path)
    solarized_image = ImageOps.solarize(image, threshold)
    solarized_image.save(output_file_path)

def vignette(input_file_path, output_file_path, param1, param2, param3):
    image = Image.open(input_file_path)
    width, height = image.size
    vignette_filter = Image.new("L", (width, height), 0)
    for x in range(width):
        for y in range(height):
            dx = x - width / 2
            dy = y - height / 2
            distance = np.sqrt(dx**2 + dy**2)
            vignette_filter.putpixel((x, y), int(255 * (1 - distance / (width / 2))))
    vignette_image = Image.composite(image, Image.new("RGB", (width, height), "black"), vignette_filter)
    vignette_image.save(output_file_path)

def pixelate(input_file_path, output_file_path, pixel_size, param1, param2):
    image = Image.open(input_file_path)
    image = image.resize(
        (image.size[0] // pixel_size, image.size[1] // pixel_size),
        Image.NEAREST
    )
    image = image.resize(
        (image.size[0] * pixel_size, image.size[1] * pixel_size),
        Image.NEAREST
    )
    image.save(output_file_path)

def apply_effect(input_file_path, output_file_path, effect_type, param1 = None, param2 = None, param3 = None):
    if effect_type == 0:    oil_painting(input_file_path, output_file_path, param1, param2, param3)
    elif effect_type == 1:  cartoon(input_file_path, output_file_path, param1, param2, param3)
    elif effect_type == 2:  grayscale(input_file_path, output_file_path, param1, param2, param3)
    elif effect_type == 3:  sepia(input_file_path, output_file_path, param1, param2, param3)
    elif effect_type == 4:  negative(input_file_path, output_file_path, param1, param2, param3)
    elif effect_type == 5:  blur(input_file_path, output_file_path, param1, param2, param3)
    elif effect_type == 6:  sharpen(input_file_path, output_file_path, param1, param2, param3)
    elif effect_type == 7:  edge_detection(input_file_path, output_file_path, param1, param2, param3)
    elif effect_type == 8:  emboss(input_file_path, output_file_path, param1, param2, param3)
    elif effect_type == 9:  brightness_adjustment(input_file_path, output_file_path, param1, param2, param3)
    elif effect_type == 10:  contrast_adjustment(input_file_path, output_file_path, param1, param2, param3)
    elif effect_type == 11:  saturation_adjustment(input_file_path, output_file_path, param1, param2, param3)
    elif effect_type == 12:  gaussian_blur(input_file_path, output_file_path, param1, param2, param3)
    elif effect_type == 13:  median_filter(input_file_path, output_file_path, param1, param2, param3)
    elif effect_type == 14:  bilateral_filter(input_file_path, output_file_path, param1, param2, param3)
    elif effect_type == 15:  thresholding(input_file_path, output_file_path, param1, param2, param3)
    elif effect_type == 16:  posterize(input_file_path, output_file_path, param1, param2, param3)
    elif effect_type == 17:  solarize(input_file_path, output_file_path, param1, param2, param3)
    elif effect_type == 18:  vignette(input_file_path, output_file_path, param1, param2, param3)
    elif effect_type == 19:  pixelate(input_file_path, output_file_path, param1, param2, param3)
if __name__ == '__main__':
   image = cv2.imread(R'C:\Users\Administrator\Documents\2.jpg')