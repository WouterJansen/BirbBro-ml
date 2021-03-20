import os
import numpy as np
import torchvision as tv
from PIL import Image
import matplotlib.pyplot as plt


IMAGES_FOLDER_IN = "images"
IMAGES_FOLDER_OUT = "images_resized"
FINAL_WIDTH = 1024
FINAL_HEIGHT = 1024


def resize_images(images_folder_in, images_folder_out, final_width, final_height):
    size = FINAL_WIDTH, FINAL_HEIGHT
    try:
        os.mkdir(IMAGES_FOLDER_OUT)
    except Exception as ex:
        pass
    for subdir, dirs, files in os.walk(images_folder_in):
        for file in files:
            im = Image.open(os.path.join(subdir, file))

            im.thumbnail(size, Image.ANTIALIAS)
            new_im = Image.new("RGB", size)
            if "png" in file:
                new_im = Image.new("RGBA", size)
            new_im.paste(im, ((size[0] - im.size[0]) // 2,
                                  (size[1] - im.size[1]) // 2))
            try:
                os.mkdir(os.path.join(IMAGES_FOLDER_OUT, subdir.split('\\')[1]))
            except Exception as ex:
                pass
            if "png" in file:
                new_im.save(os.path.join(os.path.join(images_folder_out, subdir.split('\\')[1]), file), "PNG")
            else:
                new_im.save(os.path.join(os.path.join(images_folder_out, subdir.split('\\')[1]), file), "JPEG")


def get_average_sizes(images_folder):
    # calculate image statistics (takes some time to complete)
    ds = tv.datasets.ImageFolder(images_folder)
    shapes = [(img.height, img.width) for img, _ in ds]
    heights, widths = [[h for h, _ in shapes], [w for _, w in shapes]]
    print('Average sizes:', *map(np.median, zip(*shapes)))

    # visualize the distribution of the size of images
    fig = plt.figure()
    ax = fig.add_subplot(111)
    bp = ax.boxplot([heights, widths], patch_artist=True)
    ax.set_xticklabels(['height', 'width'])
    ax.set_xlabel('image sizes')
    ax.set_ylabel('pixels')
    plt.show()


if __name__ == '__main__':
    #get_average_sizes(IMAGES_FOLDER)
    resize_images(IMAGES_FOLDER_IN, IMAGES_FOLDER_OUT, FINAL_WIDTH, FINAL_HEIGHT)