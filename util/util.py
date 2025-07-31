"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
import copy


def custom_normalization(img_HU, window=[3600, 700], rescale=True):
    # window = (window width, window level)
    img_min = window[1] - window[0] // 2
    img_max = window[1] + window[0] // 2
    img_HU[img_HU < img_min] = img_min
    img_HU[img_HU > img_max] = img_max
    img_rescaled = copy.deepcopy(img_HU).astype(np.float32)
    if rescale:
        img_rescaled[img_rescaled <= 1000] = 0.0004286 * img_rescaled[img_rescaled <= 1000] + 0.471429
        img_rescaled[img_rescaled > 1000] = 0.0000909 * img_rescaled[img_rescaled > 1000] + 0.8090909
        # plt.imshow(img_rescaled, cmap='gray')
        img_rescaled = img_rescaled * 255
        img_rescaled[img_rescaled > 255] = 255
    # img_rescaled = np.transpose(np.expand_dims(img_rescaled, 2), (2, 0, 1))
    return img_rescaled.astype(np.uint8)


def linear_normalization(img_HU):
    img_rescaled = copy.deepcopy(img_HU).astype(np.float32)
    img_rescaled = (255 * ((img_rescaled - (-1000)) / (2500 - (-1000))))
    return img_rescaled.astype(np.uint8)


def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)
