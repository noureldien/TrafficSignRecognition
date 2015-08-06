import numpy
import numpy as np
import matplotlib.pyplot as plt

import CNN.utils

import skimage
import skimage.io
from skimage import data, color
from skimage.transform import hough_circle
from skimage.feature import peak_local_max, canny
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte


def detection_proposal(img_path, min_dim=40, max_dim=60, step=1):
    # load picture and detect edges
    img = CNN.utils.rgb_to_gs(img_path)
    img_edges = canny(img, sigma=3, low_threshold=10, high_threshold=50)

    centers = []
    accums = []
    radii = []

    # detect all radii within the range
    min_radius = int(min_dim / 2)
    max_radius = int(max_dim / 2)
    hough_radii = np.arange(start=min_radius, stop=max_radius, step=step)
    hough_res = hough_circle(img_edges, hough_radii)

    for radius, h in zip(hough_radii, hough_res):
        # for each radius, extract two circles
        num_peaks = 2
        peaks = peak_local_max(h, num_peaks=num_peaks)
        centers.extend(peaks)
        accums.extend(h[peaks[:, 0], peaks[:, 1]])
        radii.extend([radius] * num_peaks)

    img_color = color.gray2rgb(img)
    img_width = img.shape[1]
    img_height = img.shape[0]

    idx_sorted = np.argsort(accums)[::-1]

    # get the sorted accumulated values
    # accums_sorted = np.asarray(accums)
    # accums_sorted = accums_sorted[idx_sorted]

    # don't consider circles with accum value less than the threshold
    accum_threshold = 0.3

    # draw the most prominent n circles, i.e those
    # with the highest n peaks
    for idx in idx_sorted:
        if accums[idx] < accum_threshold:
            continue
        center_x, center_y = centers[idx]
        radius = radii[idx]
        cx, cy = circle_perimeter(center_y, center_x, radius)
        cx[cx > img_width - 1] = img_width - 1
        cy[cy > img_height - 1] = img_height - 1
        img_color[cy, cx] = (220, 20, 20)

    # save the result
    skimage.io.imsave("D://_Dataset//GTSDB//Test_Regions//_img2.png", img_color)
