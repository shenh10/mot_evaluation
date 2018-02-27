import numpy as np

def bbox_overlap(ex_box, gt_box):
    paded_gt = np.tile(gt_box, [ex_box.shape[0],1])
    insec = intersection(ex_box, paded_gt)
    uni = areasum(ex_box, paded_gt) - insec
    return insec / uni

def intersection(a, b):
    x = np.maximum(a[:, 0], b[:, 0])
    y = np.maximum(a[:, 1], b[:, 1])
    w = np.minimum(a[:, 2], b[:, 2]) - x + 1
    h = np.minimum(a[:, 3], b[:, 3]) - y + 1
    return np.maximum(w*h,0)

def areasum(a, b):
    return (a[:, 2] - a[:,0] + 1) * (a[:, 3] - a[:,1] + 1) + (b[:, 2] - b[:, 0] + 1) * (b[:, 3] - b[:,1] + 1)

