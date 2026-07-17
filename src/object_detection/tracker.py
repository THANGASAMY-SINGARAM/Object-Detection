"""A dependency-light implementation of the SORT object tracker."""

import numpy as np
from scipy.optimize import linear_sum_assignment

def bbox_to_z(bbox):
    """
    Converts bounding box in format [x1, y1, x2, y2] to state format [u, v, s, r]^T
    where u, v is the center of the box, s is the scale (area), and r is the aspect ratio.
    """
    w = max(1e-5, bbox[2] - bbox[0])
    h = max(1e-5, bbox[3] - bbox[1])
    u = bbox[0] + w / 2.0
    v = bbox[1] + h / 2.0
    s = w * h
    r = w / float(h)
    return np.array([[u], [v], [s], [r]])

def z_to_bbox(z):
    """
    Converts state vector [u, v, s, r]^T back to bounding box format [x1, y1, x2, y2].
    """
    u, v, s, r = z[0, 0], z[1, 0], z[2, 0], z[3, 0]
    
    # Avoid zero division and negative square root
    s = max(0, s)
    r = max(1e-5, r)
    
    w = np.sqrt(s * r)
    h = np.sqrt(s / r)
    
    x1 = u - w / 2.0
    y1 = v - h / 2.0
    x2 = u + w / 2.0
    y2 = v + h / 2.0
    return np.array([x1, y1, x2, y2])

def iou(box1, box2):
    """
    Computes Intersection over Union (IoU) between two bounding boxes.
    Boxes are in format [x1, y1, x2, y2].
    """
    xx1 = max(box1[0], box2[0])
    yy1 = max(box1[1], box2[1])
    xx2 = min(box1[2], box2[2])
    yy2 = min(box1[3], box2[3])
    
    w = max(0.0, xx2 - xx1)
    h = max(0.0, yy2 - yy1)
    
    intersection = w * h
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = area1 + area2 - intersection
    if union <= 0:
        return 0.0
    return intersection / float(union)

class KalmanBoxTracker:
    """
    Represents the internal state of individual tracked objects observed as bbox.
    Uses a Kalman Filter with a constant velocity model.
    """
    count = 0
    
    def __init__(self, bbox, class_id=0):
        # State vector x: [u, v, s, r, u_dot, v_dot, s_dot]^T
        self.x = np.zeros((7, 1))
        self.x[:4] = bbox_to_z(bbox)
        
        # State transition matrix F
        self.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])
        
        # Measurement matrix H
        self.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])
        
        # Covariance matrix P
        self.P = np.diag([10.0, 10.0, 10.0, 10.0, 1000.0, 1000.0, 1000.0])
        
        # Process covariance matrix Q
        self.Q = np.diag([1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 0.0001])
        
        # Measurement covariance matrix R
        self.R = np.diag([1.0, 1.0, 10.0, 10.0])
        
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.class_id = class_id

    def predict(self):
        """
        Advances the state vector and covariance matrix using the transition model.
        """
        # If scale + velocity of scale is <= 0, clamp velocity of scale to 0
        if self.x[2, 0] + self.x[6, 0] <= 0:
            self.x[6, 0] = 0.0
            
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(z_to_bbox(self.x))
        return self.history[-1]

    def update(self, bbox, class_id=None):
        """
        Updates the state vector with a new measurement (bounding box).
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        if class_id is not None:
            self.class_id = class_id
            
        # Measurement vector z
        z = bbox_to_z(bbox)
        
        # Innovation residual y
        y = z - np.dot(self.H, self.x)
        
        # Innovation covariance S
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        
        # Kalman Gain K
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        
        # Updated state estimation x
        self.x = self.x + np.dot(K, y)
        
        # Updated covariance estimation P
        I = np.eye(7)
        self.P = np.dot(I - np.dot(K, self.H), self.P)

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return z_to_bbox(self.x)

class Sort:
    """
    Simple Online and Realtime Tracking (SORT) manager.
    """
    def __init__(self, max_age=15, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets=np.empty((0, 6))):
        """
        Params:
          dets - a numpy array of detections in the format [[x1, y1, x2, y2, score, class_id], ...]
        Requires: this method must be called once for each frame even with empty detections.
        Returns a similar array where the last elements are the object ID and class ID.
        """
        self.frame_count += 1
        
        # Get predicted positions from existing trackers
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
                
        # Remove trackers that produced NaN predictions
        for index in sorted(to_del, reverse=True):
            self.trackers.pop(index)
            
        # Re-evaluate remaining trackers list size
        trks = np.delete(trks, to_del, axis=0)
        
        # Match detections to track predictions
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            dets, trks, self.trackers, self.iou_threshold
        )
        
        # Update matched trackers
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :4], class_id=dets[m[0], 5])
            
        # Create and initialize new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :4], class_id=dets[i, 5])
            self.trackers.append(trk)
            
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                # Format: [x1, y1, x2, y2, track_id, class_id]
                ret.append(np.concatenate((d, [trk.id + 1, trk.class_id])))
            i -= 1
            # Remove dead tracklet
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
                
        if len(ret) > 0:
            return np.stack(ret)
        return np.empty((0, 6))

def associate_detections_to_trackers(detections, trackers, tracker_objects, iou_threshold=0.3):
    """
    Assigns detections to tracked object predictions using Hungarian algorithm.
    """
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 2), dtype=int)
        
    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
    
    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            # Enforce same-class tracking if class details are available
            if det[5] != tracker_objects[t].class_id:
                iou_matrix[d, t] = 0.0
            else:
                iou_matrix[d, t] = iou(det[:4], trk[:4])
                
    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            # Quick assignment if matches are unique and non-overlapping
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            # General Hungarian algorithm assignment
            matched_indices = np.stack(linear_sum_assignment(-iou_matrix), axis=1)
    else:
        matched_indices = np.empty((0, 2), dtype=int)
        
    unmatched_detections = []
    for d in range(len(detections)):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
            
    unmatched_trackers = []
    for t in range(len(trackers)):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)
            
    # Filter out matches with low IoU
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
            
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)
        
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)
