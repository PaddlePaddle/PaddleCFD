import numpy as np
from PIL import Image


def concat_images_square(images, padding):
    columns = int(np.ceil(np.sqrt(len(images))))
    rows = int(np.ceil(len(images) / columns))

    w, h = images[0].size
    concated = Image.new(images[0].mode, (w * columns + padding * (columns - 1), h * rows + padding * (rows - 1)))
    for i in range(len(images)):
        col = (i % columns)
        row = i // columns
        concated.paste(images[i], (w * col + padding * (col - 1), h * row + padding * (row - 1)))
    return concated
