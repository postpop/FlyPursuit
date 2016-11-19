import numpy as np
import cv2
from cv2to3.cv2to3 import *
import scipy.ndimage as sci


def circular_kernel(kernel_size=3):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))


def dilate(frame, kernel_size=3):
    return cv2.dilate(frame, circular_kernel(kernel_size))


def erode(frame, kernel_size=3):
    return cv2.erode(frame, circular_kernel(kernel_size))


def close(frame, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel)


def threshold(frame, threshold):
    return frame > threshold


def crop(frame, coords):
    """crop x,y"""
    if len(frame.shape) == 3:
        return frame[coords[1]:coords[3], coords[0]:coords[2], :]
    elif len(frame.shape)==2:
        return frame[coords[1]:coords[3], coords[0]:coords[2]]


def clean_labels(labeled_frame, new_labels=None, force_cont=False):
    """maps unique values in `labeled_frame` to new values from `new_labels`
    `force_cont` ensures that new labels will be continuous, starting at 0
    source: http://stackoverflow.com/questions/13572448/change-values-in-a-numpy-array
    """
    old_labels = np.unique(labeled_frame)
    # new_labels gives the new values you wish old_labels to be mapped to.
    if new_labels is None:
        new_labels = np.array(range(old_labels.shape[0]))

    # ensure labels are sorted - do this always since it should be cheap
    sort_idx = np.argsort(old_labels)
    old_labels = old_labels[sort_idx]
    new_labels = new_labels[sort_idx]
    # remap
    index = np.digitize(labeled_frame, old_labels, right=True)
    labeled_frame = new_labels[index]
    if force_cont:
        labeled_frame, new_labels, _ = clean_labels(labeled_frame)
    return labeled_frame, new_labels, old_labels


def getpoints(frame):
    """get x,y indices of foreground pixels"""
    # maybe implement in
    # cython: http://stackoverflow.com/questions/26222674/python-cython-numpy-optimization-of-np-nonzero
    # or numba?
    points = np.nonzero(frame)  # returns 2 lists - one for row, one for columns indices
    return np.vstack(points).astype(np.float32).T  # convert to Nx2 array


def segment_connected_components(frame):
    """get properties of all connected components"""
    labeled_frame, nlbl = sci.label(frame)
    points = getpoints(frame)
    # get labels for points
    columns = np.array(points[:, 1], dtype=np.intp)
    rows = np.array(points[:, 0], dtype=np.intp)
    labels = labeled_frame[rows, columns]
    # get centers etc
    centers = np.array(sci.center_of_mass(frame,labeled_frame, range(1, nlbl + 1)), dtype=np.uint)  # convert to list from tuple
    size = sci.labeled_comprehension(frame, labeled_frame, range(1, nlbl + 1), np.alen, out_dtype=np.uint, default=0, pass_positions=False)
    std = sci.labeled_comprehension(frame, labeled_frame, range(1, nlbl + 1), np.std, out_dtype=np.uint, default=0, pass_positions=False)
    return centers, labels, points, std, size, labeled_frame


def segment_center_of_mass(frame):
    """ get center of mass of all points - usually very robust for single-fly tracking since it ignores small specks"""
    points = getpoints(frame)
    size     = points.shape[0] 
    centers = np.mean(points, axis=0)
    std      = np.std(points, axis=0)
    labels  = np.ones(points.shape, dtype=np.uint)  # for compatibility with multi-object methods
    return centers, labels, points, std, size


def segment_cluster(frame, num_clusters=1, term_crit=(cv2.TERM_CRITERIA_EPS, 30, 0.1), init_method=cv2.KMEANS_PP_CENTERS):
    """cluster points to get fly positions"""
    points = getpoints(frame)
    cluster_compactness, labels, centers = cv2.kmeans(points, num_clusters, None, term_crit, 20, init_method)
    return centers, labels, points, cluster_compactness


def get_chambers(background, chamber_threshold=0.6, min_size=40000, max_size=50000):
    """detect (bright) chambers in background"""
    if len(background.shape) > 2:
        background = background[:, :, 0]
    background_thres = np.double(threshold(background, chamber_threshold * np.mean(background)))
    background_thres = close(background_thres, kernel_size=11)
    background_thres = dilate(background_thres, kernel_size=11)
    # initial guess of chambers
    _, _, _, _, area, labeled_frame = segment_connected_components(background_thres)
    area = np.insert(area, 0, 0)  # add dummy valye for the area of the background, which is not returned by `segment_connected_components`
    # weed out too small chambers based on area
    unique_labels = np.unique(labeled_frame)
    condlist = np.any([area < min_size, area > max_size], axis=0)
    unique_labels[condlist] = 0
    labeled_frame, _, _ = clean_labels(labeled_frame, unique_labels, force_cont=True)
    return labeled_frame


def get_bounding_box(labeled_frame):
    """get bounding boxes of all components"""

    uni_labels = np.unique(labeled_frame)
    bounding_box = np.ndarray( ( np.max(uni_labels)+1, 2, 2), dtype=np.int )
    for ii in range(uni_labels.shape[0]):
        points = getpoints(labeled_frame==uni_labels[ii]);
        bounding_box[uni_labels[ii],:,:] = np.vstack( (np.min(points, axis=0), np.max(points, axis=0)) )
    return bounding_box


def annotate(frame, centers=None, lines=None):
    """annotate frame"""
    if centers is not None:
        for center in centers:
            cv2.circle(frame, (center[1], center[0]), radius=6, color=[0, 0, 255], thickness=2)
    if lines is not None:
        for line in lines:
            cv2.line(frame, tuple(line[0]), tuple(line[1]), color=[0, 0, 255], thickness=2)
    return frame


def show(frame, window_name="frame", time_out=1):
    """display frame"""
    cv2.imshow(window_name, np.float32(frame))
    cv2.waitKey(time_out)


def test():
    frame = cv2.imread("test/frame.png")
    background = cv2.imread("test/background.png")

    foreground = (background.astype(np.float32) - frame) / 255.0
    show(foreground, time_out=100)

    foreground = dilate(foreground, 3)
    show(foreground, time_out=100)

    foreground = erode(foreground, 3)
    show(foreground, time_out=100)

    foreground_thres = threshold(foreground, 0.6)
    foreground_thres = foreground_thres[:, :, 0]
    show(foreground_thres, time_out=100)

    centers, labels, _, _, area, labeled_frame = segment_connected_components(foreground_thres)
    print(centers)
    print(area)
    show(labeled_frame, time_out=1000)
    show(frame / 255, centers, time_out=2000)

    centers, _, _, _ = segment_cluster(foreground_thres, 12)
    print(centers)

    center_of_mass, _, _, _, _ = segment_center_of_mass(foreground_thres)
    print(center_of_mass)

    labeled_frame = get_chambers(background)
    bounding_box = get_bounding_box(labeled_frame)
    show(labeled_frame / np.max(labeled_frame), time_out=1000)
    show(labeled_frame == 5, time_out=1000)  # pick chamber #5

    show(crop(frame, [10, 550, 100,-1])/255, time_out=1000)

    labeled_frame = get_chambers(background)
    labeled_frame[labeled_frame == 10] = 0
    labeled_frame[labeled_frame == 5] = 20

    show(labeled_frame / np.max(labeled_frame), time_out=1000)
    new_labels = np.unique(labeled_frame)
    new_labels[-1] = 0  # delete last chamber (#20)
    # delete last chamber only
    print(np.unique(clean_labels(labeled_frame, new_labels)[0]))
    # remap to cont labels
    print(np.unique(clean_labels(labeled_frame, new_labels, force_cont=True)[0]))
    show(labeled_frame / np.max(labeled_frame), time_out=1000)


if __name__ == "__main__":
    test()
    