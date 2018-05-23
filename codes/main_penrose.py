"""
Main - Detection-to-Tracking & Results Display
@author: Pengluo Wang
"""

import time
import copy
import numpy as np
import matplotlib.pyplot as plt
import glob
from moviepy.editor import VideoFileClip
from collections import deque
from sklearn.utils.linear_assignment_ import linear_assignment

import helpers
import detector
import tracker

# global variables
DEBUG_MODE = 0
PROC_MODE = 1
# debug mode:
#   0: video pipeline processing
#   1: frames pipeline processing
# process mode:
#   1: detection results (measurement z)
#   2: prediction results (prior state estimate x_hat_super_minus, x_hsm)
#   3: tracking results (posterior state estimate x_hat) (DEMO)
frame_idx = 0           # current frame No.
tracker_list = []       # list of trackers
tracker_id_list = deque(range(0, 100))   # list for track ID - 100 different object
min_hits = 3
max_losses = 3


def assign_detections_to_trackers(trackers, detections, IoU_threshold=0.3):
    """
    Assign detections to trackers object (both represented as bounding boxes)
    :param trackers: list of bounding boxes (x_boxes)
    :param detections: list of bounding boxes (z_boxes)
    :param IoU_threshold:
    :return: 3 lists of matches, unmatched_detections, and unmatched_trackers
    """

    # create IoU matrix
    IoU_matrix = np.zeros((len(trackers), len(detections)), dtype=np.float32)
    for t, trk in enumerate(trackers):
        for d, det in enumerate(detections):
            IoU_matrix[t, d] = helpers.IoU(trk, det)

    # using Hungarian (Munkres) algorithm to solve linear assignment problem
    matched_indices = linear_assignment(- IoU_matrix)

    # unmatched_detections
    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 1]:
            unmatched_detections.append(d)

    # unmatched_trackers
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 0]:
            unmatched_trackers.append(t)

    # matches
    matches = []
    for m in matched_indices:
        if IoU_matrix[m[0], m[1]] < IoU_threshold:
            unmatched_trackers.append(m[0])
            unmatched_detections.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


def pipeline(image):
    """
    Pipeline function for object detection and tracking process
    :param image: frame to be processed
    :return: processed frame
    """

    # global variables
    global frame_idx
    global tracker_list
    global tracker_id_list
    global min_hits
    global max_losses
    global DEBUG_MODE
    global PROC_MODE

    image_h, image_w, _ = image.shape

    frame_idx += 1
    if DEBUG_MODE:
        print('\nFrame %03d:' % frame_idx)

    # generate location boxes
    z_boxes = detect.get_localization(image)

    # read previous location boxes for assigning detections to trackers
    x_boxes = []
    if len(tracker_list) > 0:
        for trk in tracker_list:
            x_boxes.append(trk.box)

    matched, unmatched_detections, unmatched_trackers = \
        assign_detections_to_trackers(x_boxes, z_boxes, IoU_threshold=0.2)

    # tracking on matched objects
    if matched.size > 0:
        for trk_idx, det_idx in matched:
            z = z_boxes[det_idx]                # Kalman measurement z(k)
            #z = np.expand_dims(z, axis=0).T     # change shape from (4,) to (4, 1)
            trk_tmp = tracker_list[trk_idx]
            trk_tmp.kalman_filter(z)
            x_boxes[trk_idx] = trk_tmp.box
            trk_tmp.hits += 1
            if trk_tmp.hits >= min_hits:
                trk_tmp.good_tracker = 1
            trk_tmp.losses = 0

    # create tracker on unmatched detections
    if len(unmatched_detections) > 0:
        for idx in unmatched_detections:
            z = z_boxes[idx]
            #z = np.expand_dims(z, axis=0).T
            id = tracker_id_list.popleft()      # assign an ID for the tracker
            trk_tmp = tracker.Tracker(id, image_h, image_w)     # create a new tracker
            #x = np.array([[z[0], 0, z[1], 0, z[2], 0, z[3], 0]]).T
            x = np.array([z[0], 0, z[1], 0, z[2], 0, z[3], 0])
            trk_tmp.x_state = x
            trk_tmp.predict_only()
            tracker_list.append(trk_tmp)    # put into tracker list
            x_boxes.append(trk_tmp.box)

    # tracking on unmatched trackers
    if len(unmatched_trackers) > 0:
        for idx in unmatched_trackers:
            trk_tmp = tracker_list[idx]
            trk_tmp.losses += 1
            trk_tmp.hits = 0
            trk_tmp.predict_only()
            x_boxes[idx] = trk_tmp.box

    # delete lost tracker
    lost_tracker = filter(lambda x: x.is_lost(max_losses), tracker_list)
    for trk in lost_tracker:
        tracker_id_list.append(trk.id)
    tracker_list = [x for x in tracker_list if not x.is_lost(max_losses)]

    print('No. of tracker: %2d' % len(tracker_list))

    if DEBUG_MODE == 1:
        image_det = copy.copy(image)
        for idx in range(len(z_boxes)):
            image_det = helpers.draw_box_label(image_det, z_boxes[idx], box_color=(255, 0, 0), show_label=False)
        plt.imshow(image_det)
        plt.title('Detection result')
        plt.axis('off')
        plt.show()

        image_pred = copy.copy(image)
        for trk in tracker_list:
            image_pred = helpers.draw_box_label(image_pred, trk.box, box_color=(0, 255, 0), id=trk.id)
        plt.imshow(image_pred)
        plt.title('Prediction result')
        plt.axis('off')
        plt.show()

        image_trk = copy.copy(image)
        for trk in tracker_list:
            if trk.hits >= min_hits or (trk.hits > 0 and trk.good_tracker):
                image_trk = helpers.draw_box_label(image_trk, trk.box, box_color=(0, 0, 255), id=trk.id)
        plt.imshow(image_trk)
        plt.title('Tracking result')
        plt.axis('off')
        plt.show()

    if PROC_MODE == 1:
        image_det = image
        for idx in range(len(z_boxes)):
            image_det = helpers.draw_box_label(image_det, z_boxes[idx], box_color=(255, 0, 0), show_label=False)
        return image_det

    elif PROC_MODE == 2:
        image_pred = image
        for trk in tracker_list:
            image_pred = helpers.draw_box_label(image_pred, trk.box, box_color=(0, 0, 255), id=trk.id)

        return image_pred

    elif PROC_MODE == 3:
        image_trk = image
        for trk in tracker_list:
            if trk.hits >= min_hits or (trk.hits > 0 and trk.good_tracker):
                image_trk = helpers.draw_box_label(image_trk, trk.box, box_color=(0, 255, 0), id=trk.id)

        return image_trk
    else:
        ValueError('PROC_MODE must be 1, 2, or 3.')


# Main
if __name__ == "__main__":
    detect = detector.CarDetector()

    if DEBUG_MODE:      # process on consecutive frames
        images = [plt.imread(f) for f in sorted(glob.glob('./part2/*.jpg'))]

        for idx in range(len(images)):
            img = images[idx]
            pipeline(img)

    else:               # process on video
        start = time.time()
        clip = VideoFileClip("video.mp4")
        clip_proc = clip.fl_image(pipeline)
        if PROC_MODE == 1:
            clip_proc.write_videofile('result_detection.mp4', audio=False)
        elif PROC_MODE == 2:
            clip_proc.write_videofile('result_prediction.mp4', audio=False)
        elif PROC_MODE == 3:
            clip_proc.write_videofile('result_tracking.mp4', audio=False)
        else:
            ValueError('PROC_MODE must be 1, 2, or 3.')
        end = time.time()
        print('Time usage: %.1f s' % (end - start))
