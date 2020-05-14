"basics imports"
from __future__ import print_function
import warnings

warnings.filterwarnings('ignore')
import numpy as np
from datetime import datetime as dt

"Tracks"
from tracking.kalman_filters import KalmanBoxTracker

"High performance compiler"
from numba import jit

"LAP solvers"
# import lapjv
import lapsolver
from sklearn.utils.linear_assignment_ import linear_assignment

"Operations for Matrix"
@jit(nopython=True)
def distance(bb_test, bb_gt):
    """
    Computes IUO between two bboxes in the form [x1,y1,x2,y2]
    """
    xxd = 0.5 * (bb_test[0] + bb_test[2])
    yyd = 0.5 * (bb_test[1] + bb_test[3])
    xxt = 0.5 * (bb_gt[0] + bb_gt[2])
    yyt = 0.5 * (bb_gt[1] + bb_gt[3])
    d = (xxd - xxt) ** 2 + (yyd - yyt) ** 2
    return (np.sqrt(d))

@jit(nopython=True)
def iou(bb_test, bb_gt):
    """
    Computes IUO between two bboxes in the form [x1,y1,x2,y2]
    Relacion porcentual del Area de intercepcion sobre
    el area de union de los dos boxes. valores entre [0,1]
    """
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
              + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
    return (o)

"LAP"
def LAP_iou_sklearn(detections, trackers, threshold):
    """
    Assigns detections to tracked object (both represented as bounding boxes)
    Hungarian algorithm

    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    # todo delete line
    val = False
    if (len(trackers) == 0):
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = iou(det, trk)

    matched_indices = linear_assignment(-iou_matrix)  # [row,col]

    unmatched_detections = []
    for d, det in enumerate(detections):
        if (d not in matched_indices[:, 0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if (t not in matched_indices[:, 1]):
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if (iou_matrix[m[0], m[1]] < threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if (len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

def LAP_dist_sklearn(detections, trackers, threshold):
    """
    Assigns detections to tracked object (both represented as bounding boxes)
    Hungarian algorithm

    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if (len(trackers) == 0):
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

    distance_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)

    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            distance_matrix[d, t] = distance(det, trk)
    matched_indices = linear_assignment(distance_matrix)  # [row,col]
    val = matched_indices

    unmatched_detections = []
    for d, det in enumerate(detections):
        if (d not in matched_indices[:, 0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if (t not in matched_indices[:, 1]):
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    # try:
    for m in matched_indices:
        if (distance_matrix[m[0], m[1]] > threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if (len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

def LAP_iou_lapsolver_solvedense(detections, trackers, threshold):
    """
    Assigns detections to tracked object (both represented as bounding boxes)
    Hungarian algorithm

    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if (len(trackers) == 0):
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)

    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = iou(det, trk)
    matched_indices = lapsolver.solve_dense(-iou_matrix)

    unmatched_detections = []
    for d, det in enumerate(detections):
        if (d not in matched_indices[0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if (t not in matched_indices[1]):
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    # try:
    for m in zip(matched_indices[0], matched_indices[1]):
        if (iou_matrix[m[0], m[1]] < threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(np.array(m).reshape(1, 2))
    if (len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

def LAP_dist_lapsolver_solvedense(detections, trackers, threshold):
    """
    Assigns detections to tracked object (both represented as bounding boxes)
    Hungarian algorithm

    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if (len(trackers) == 0):
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

    distance_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)

    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            distance_matrix[d, t] = distance(det, trk)
    matched_indices = lapsolver.solve_dense(distance_matrix)

    unmatched_detections = []
    for d, det in enumerate(detections):
        if (d not in matched_indices[0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if (t not in matched_indices[1]):
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    # try:
    for m in zip(matched_indices[0], matched_indices[1]):
        if (distance_matrix[m[0], m[1]] > threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(np.array(m).reshape(1, 2))
    if (len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

class Sort(object):
    def __init__(self
                 , threshold
                 , matrix_type
                 , LAP_type
                 , max_age=3
                 , min_hits=3
                 ):
        if matrix_type=='DIST':
            if threshold <=1:
                print("revisar threshold para DIST");
                exit()
        if matrix_type == 'IOU':
            if threshold <0 or threshold > 1:
                print("revisar threshold para IOU");exit()

        self.max_age                        = max_age
        self.min_hits                       = min_hits
        self.Matrix_type                    = matrix_type
        self.threshold                      = threshold
        self.LAP_type                       = LAP_type
        self.frame_count                    = 0

    def update(self
               , dets
               , tracked_detections):


        if dets is None:
            return None

        erased_trackers = []
        to_del = []

        dets = np.array(dets)
        self.frame_count += 1

        # get predicted locations from existing trackers.
        trks = np.zeros((len(tracked_detections), 5))


        for t, trk in enumerate(trks):
            pos = tracked_detections[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if (np.any(np.isnan(pos))):
                to_del.append(t)

        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))

        for t in reversed(to_del):
            trk = tracked_detections.pop(t)
            trk.set_end_time(dt.now())
            erased_trackers.append(trk)

        if self.Matrix_type == "IOU":
            if self.LAP_type == "sklearn_lap":
                matched, unmatched_dets, unmatched_trks = LAP_iou_sklearn(dets, trks, self.threshold)
            elif self.LAP_type == "lapsolver_solvedense":
                matched, unmatched_dets, unmatched_trks = LAP_iou_lapsolver_solvedense(dets, trks, self.threshold)
            else:
                print("No LAP_type Found")
        elif self.Matrix_type == "DIST":
            if self.LAP_type == "sklearn_lap":
                matched, unmatched_dets, unmatched_trks = LAP_dist_sklearn(dets, trks, self.threshold)
            elif self.LAP_type == "lapsolver_solvedense":
                matched, unmatched_dets, unmatched_trks = LAP_dist_lapsolver_solvedense(dets, trks, self.threshold)
            else:
                print("No LAP_type Found")
        else:
            print("No MAtrix_type Found")
            exit()

        # update matched trackers with assigned detections
        for t, trk in enumerate(tracked_detections):
            if (t not in unmatched_trks):
                d = matched[np.where(matched[:, 1] == t)[0], 0]
                trk.update(dets[d][0][:4])

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i][:],dt.now())
            tracked_detections.append(trk)

        # Evaluate actual trackers. Eliminate trackers that dont fit the parameter time_since_update
        i = len(tracked_detections)
        for trk in reversed(tracked_detections):
            i -= 1
            if (trk.time_since_update > self.max_age):
                trk = tracked_detections.pop(i)
                trk.set_end_time(dt.now())
                erased_trackers.append(trk)

        return tracked_detections, erased_trackers