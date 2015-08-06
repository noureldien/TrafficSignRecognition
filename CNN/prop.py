import numpy
import numpy as np
import matplotlib.pyplot as plt

import cv2

import CNN.utils
import CNN.nms

import skimage
import skimage.io
import skimage.exposure
from skimage import data, color
from skimage.transform import hough_circle
from skimage.feature import peak_local_max, canny
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte


def detection_proposal_and_save(img_path, min_dim=40, max_dim=160):
    # extract detection proposals for circle-based traffic signs
    # suppress the extracted circles to weak and strong regions
    # draw the regions on the image and save it

    # load picture and detect edges
    img_color = cv2.imread(img_path)
    img = cv2.cvtColor(img_color, cv2.COLOR_BGRA2GRAY)
    img = skimage.exposure.equalize_hist(img)
    img = skimage.exposure.rescale_intensity(img, in_range=(0.2, 0.75))

    regions_weak, regions_strong, map = detection_proposal(img, min_dim, max_dim)

    # draw and save the result, for testing purposes
    red_color = (255, 0, 0)
    yellow_color = (255, 212, 84)
    for loc in regions_weak:
        cv2.rectangle(img_color, (loc[0], loc[1]), (loc[2], loc[3]), yellow_color, 1)
    for loc in regions_strong:
        cv2.rectangle(img_color, (loc[0], loc[1]), (loc[2], loc[3]), red_color, 2)
    # save the result
    skimage.io.imsave("D://_Dataset//GTSDB//Test_Regions//_img2.png", img_color)


def detection_proposal(img_preprocessed, min_dim, max_dim):
    img = img_preprocessed
    img_edge = canny(img * 255, sigma=3, low_threshold=10, high_threshold=50)

    centers = []
    accums = []
    radii = []
    regions = []

    # detect all radii within the range
    min_radius = int(min_dim / 2)
    max_radius = int(max_dim / 2)
    hough_radii = np.arange(start=min_radius, stop=max_radius, step=1)
    hough_res = hough_circle(img_edge, hough_radii)

    for radius, h in zip(hough_radii, hough_res):
        # for each radius, extract 2 circles
        num_peaks = 2
        peaks = peak_local_max(h, num_peaks=num_peaks)
        centers.extend(peaks)
        accums.extend(h[peaks[:, 0], peaks[:, 1]])
        radii.extend([radius] * num_peaks)

    # don't consider circles with accum value less than the threshold
    idx_sorted = np.argsort(accums)[::-1]
    accum_threshold = 0.3
    for idx in idx_sorted:
        idx = int(idx)
        if accums[idx] < accum_threshold:
            continue
        center_y, center_x = centers[idx]
        radius = radii[idx]
        x1 = center_x - radius
        y1 = center_y - radius
        x2 = center_x + radius
        y2 = center_y + radius

        # we don't want to collect circles, but we want to collect regions
        # later on, we'll suppress these regions
        regions.append([x1, y1, x2, y2])

    # suppress the regions to extract the strongest ones
    if len(regions) > 0:
        min_overlap = 3
        overlap_thresh = 0.7
        regions_weak, regions_strong = CNN.nms.non_max_suppression_accurate(boxes=regions, overlap_thresh=overlap_thresh, min_overlap=min_overlap)

        # create binary map using only the strong regions
        img_shape = img_preprocessed.shape
        map = np.zeros(shape=(img_shape[0], img_shape[1]), dtype=bool)
        for r in regions_strong:
            map[r[1]:r[3], r[0]:r[2]] = True
    else:
        regions_weak = []
        regions_strong = []
        map = []

    return regions_weak, regions_strong, map


def __hough_circle_detection(img_path, min_dim=40, max_dim=60, step=1):
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
        # for each radius, extract 2 circles
        num_peaks = 2
        peaks = peak_local_max(h, num_peaks=num_peaks)
        centers.extend(peaks)
        accums.extend(h[peaks[:, 0], peaks[:, 1]])
        radii.extend([radius] * num_peaks)

    img_width = img.shape[1]
    img_height = img.shape[0]

    # get the sorted accumulated values
    # accums_sorted = np.asarray(accums)
    # accums_sorted = accums_sorted[idx_sorted]

    # don't consider circles with accum value less than the threshold
    accum_threshold = 0.3
    idx_sorted = np.argsort(accums)[::-1]

    # draw the most prominent n circles, i.e those
    # with the highest n peaks
    img_color = color.gray2rgb(img)
    for idx in idx_sorted:
        if accums[idx] < accum_threshold:
            continue
        center_y, center_x = centers[idx]
        radius = radii[idx]
        cx, cy = circle_perimeter(center_x, center_y, radius)
        cx[cx > img_width - 1] = img_width - 1
        cy[cy > img_height - 1] = img_height - 1
        img_color[cy, cx] = (220, 20, 20)

    # save the result
    skimage.io.imsave("D://_Dataset//GTSDB//Test_Regions//_img2_.png", img_color)
