# import the necessary packages
import numpy as np

# Malisiewicz et al.
def non_max_suppression(boxes, overlap_thresh, min_overlap):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # cast as numpy if needed
    if isinstance(boxes, list):
        boxes = np.asarray(boxes)

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []
    strong_pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        ii = idxs[:last]

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[ii])
        yy1 = np.maximum(y1[i], y1[ii])
        xx2 = np.minimum(x2[i], x2[ii])
        yy2 = np.minimum(y2[i], y2[ii])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        deleted_idx = np.concatenate(([last], np.where(overlap > overlap_thresh)[0]))
        idxs = np.delete(idxs, deleted_idx)

        # for strong suppression, only pick regions with many neighbours
        if deleted_idx.shape[0] >= min_overlap:
            strong_pick.append(i)
        else:
            pick.append(i)

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int"), boxes[strong_pick].astype("int")


# Malisiewicz et al. with mean-suppression
def non_max_suppression_accurate(boxes, overlap_thresh, min_overlap):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # cast as numpy if needed
    if isinstance(boxes, list):
        boxes = np.asarray(boxes)

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []
    strong_pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        ii = idxs[:last]

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[ii])
        yy1 = np.maximum(y1[i], y1[ii])
        xx2 = np.minimum(x2[i], x2[ii])
        yy2 = np.minimum(y2[i], y2[ii])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # get the indexes that need to be deleted
        deleted_i = np.concatenate(([last], np.where(overlap > overlap_thresh)[0]))

        # instead of picking up the base box of the suppressed boxes
        # we might want instead to pick up their means
        mean_box = np.mean(boxes[idxs[deleted_i]], axis=0)

        # for if we have many neighbours, then it's strong suppression
        # else, it's week one
        if deleted_i.shape[0] >= min_overlap:
            strong_pick.append(mean_box)
        else:
            pick.append(mean_box)

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, deleted_i)

    # return only the bounding boxes that were picked using the
    # integer data type
    return np.asarray(pick, dtype=int), np.asarray(strong_pick, dtype=int)
