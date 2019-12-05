import os
import time
import logging
import operator
import numpy as np
import coloredlogs
from PIL import Image

import Utils
from Utils.shortcuts import channel_last

from matplotlib import pyplot as plt

plt.switch_backend("Agg")


def plot_image(images, name, shape=None, figsize=(10, 10)):
    images = [np.minimum(np.maximum(img, 0.0), 1.0) for img in images]
    images = channel_last(images, is_array=True)

    len_list = len(images)
    im_list = []
    for i in range(len_list):
        im_list.append(images[i])

    if shape is None:
        unit = int(len_list**0.5)
        shape = (unit, unit)

    imshape = im_list[0].shape
    if imshape[2] == 1:
        im_list = [np.repeat(im, 3, axis=2) for im in im_list]
    else:
        im_list = [im for im in im_list]

    fig, axes = plt.subplots(nrows=shape[1], ncols=shape[1], figsize=figsize)
    for idx, image in enumerate(im_list):
        row = idx // shape[0]
        col = idx % shape[1]
        axes[row, col].axis("off")
        axes[row, col].imshow(image, cmap="gray", aspect="auto")
    plt.subplots_adjust(wspace=.05, hspace=.05)
    plt.tight_layout()
    plt.savefig(name)
    plt.close(fig)


def pile_image(inputs, name, shape=None, path=None, subfolder=True):
    inputs = channel_last(inputs, is_array=True)
    if isinstance(inputs, (list, tuple)):
        list_num = len(inputs)
        len_list = inputs[0].shape[0]
        im_list = []
        for i in range(len_list):
            temp_im = []
            for j in range(list_num):
                temp_im.append(inputs[j][i])
            temp_im = np.concatenate(temp_im, axis=1)
            im_list.append(temp_im)
    else:
        len_list = inputs.shape[0]
        im_list = []
        for i in range(len_list):
            im_list.append(inputs[i])

    imshape = im_list[0].shape
    if imshape[2] == 1:
        im_list = [np.repeat(im, 3, axis=2) for im in im_list]
        imshape = im_list[0].shape[:2]
    else:
        im_list = [im for im in im_list]
        imshape = im_list[0].shape[:2]

    len_list = len(im_list)
    if shape is None:
        unit = int(len_list**0.5)
        shape = (unit, unit)
    size = (shape[0] * imshape[0], shape[1] * imshape[1])
    result = Image.new("RGB", size)
    for i in range(min(len_list, shape[0] * shape[1])):
        x = i // shape[0] * imshape[0]
        y = i % shape[1] * imshape[1]
        temp_im = Image.fromarray(im_list[i])
        result.paste(temp_im, (x, y))

    if path is None:
        logging.critical("Not saving any images, just return.")
        return result

    try:
        result.save(os.path.join(path, name))
    except IOError:
        logging.critical("Unable to Save Images!")
    return result
