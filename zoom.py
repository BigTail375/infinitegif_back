from PIL import Image
from numpy import average, array, exp, sqrt

def image2recrusive(image, REC_COUNT = 3, STEPS = 50, DURATION = 20, MODE = "linear"):
    # define varaibles
    INNER_WIDTH = image.width // 10
    INNER_HEIGHT = image.height // 10
    INNER_X_LEFT = image.width // 2 - INNER_WIDTH // 2
    INNER_Y_TOP = image.height // 2 - INNER_HEIGHT // 2
    GIF_WIDTH = image.width // 3
    GIF_HEIGHT = image.height // 3

    # generate source image
    for i in range(REC_COUNT):
        small_image = image.resize((INNER_WIDTH, INNER_HEIGHT), Image.LANCZOS)
        image.paste(small_image, (INNER_X_LEFT, INNER_Y_TOP))
    
    cropend = array([INNER_X_LEFT, INNER_Y_TOP, INNER_X_LEFT + INNER_WIDTH, INNER_Y_TOP + INNER_HEIGHT])
    cropstart = array([0, 0, image.size[0], image.size[1]])
    frames = []

    for i in range(STEPS):
        mode = {"linear": [STEPS-i, i], "quad": [(STEPS-i)**2, i**2], "sqrt":[sqrt(STEPS-i), sqrt(i)]}
        # calculate new crop
        new_dim = average([cropstart] + [cropend], weights=mode[MODE], axis = 0)
        # append cropped, rescaled frame to gif
        frames.append(image.crop(new_dim).resize((GIF_WIDTH, GIF_HEIGHT)))
    return frames

# # modify these params
# INNER_SIZE = 360
# INNER_X_LEFT = 1370
# INNER_Y_TOP = 1290
# STEPS = 50
# GIF_RES = 1000
# DURATION = 20
# MODE = "linear" # or quad or sqrt
# IMAGE = "zoom1.png"

# init helper data structures
# image = Image.open(IMAGE)

# cropend = array([INNER_X_LEFT, INNER_Y_TOP, INNER_X_LEFT+INNER_SIZE, INNER_Y_TOP+INNER_SIZE])
# cropstart = array([0, 0, image.size[0], image.size[1]])
# frames = []

# for i in range(STEPS+1):
#     mode = {"linear": [STEPS-i, i], "quad": [(STEPS-i)**2, i**2], "sqrt":[sqrt(STEPS-i), sqrt(i)]}
#     # calculate new crop
#     new_dim = average([cropstart] + [cropend], weights=mode[MODE], axis = 0)
#     # append cropped, rescaled frame to gif
#     frames.append(image.crop(new_dim).resize((GIF_RES, GIF_RES)))

if __name__ == '__main__':
    image = Image.open('1.jpg')
    frames = image2recrusive(image)
    # save output
    frames[0].save('1.gif', format='GIF', append_images=frames[1:], save_all=True, duration=10, loop=0)