import argparse
import math
import os

import cv2
import numpy as np
from cv2 import NORM_HAMMING
from ultralytics import YOLO

model = YOLO("YOLOV8WinterSummer.pt")

parser = argparse.ArgumentParser(description="Arguments to give",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("filepath", help="path to the video file")
parser.add_argument("video", help="make result video", nargs='?', default="False")
args = parser.parse_args()
make_video = args.video
if make_video == "False":
    make_video = False
if make_video == "True":
    make_video = True
cap = cv2.VideoCapture(args.filepath)
fps = cap.get(cv2.CAP_PROP_FPS)
folder_name = args.filepath.split("/")[-1]
if not os.path.exists('output'):
    os.makedirs('output')

if not os.path.exists(f'output/{folder_name}'):
    os.makedirs(f'output/{folder_name}')
file_name = args.filepath.split("/")[-1][:19]
is_first_frame = True
sift = cv2.ORB_create(500, edgeThreshold=0, nlevels=20)
tracker = {}
translation_history = []
direction_history = []
history_size = 5
direction_size = 50
id_generator = 0
border_life = 25
distance_arr = []
trees = {}
all_time_smallest_distance = 999999
all_time_biggest_distance = -1


def show_results(tracker, frame, image_size=(640, 640), font_scale=0.002, thickness_scale=0.002):
    """
           Add the bounding boxes to the frame for visualization.
    """
    for key, (x1, y1, x2, y2, id, life_counter, wasDetected) in tracker.items():
        if wasDetected:
            color = (0, 255, 0)
        else:
            color = (255, 0, 0)

        scaling_factor_width = frame.shape[1] / image_size[0]
        scaling_factor_height = frame.shape[0] / image_size[1]

        x1 = int(x1 * scaling_factor_width)
        y1 = int(y1 * scaling_factor_height)
        x2 = int(x2 * scaling_factor_width)
        y2 = int(y2 * scaling_factor_height)

        font_size = min(frame.shape[1], frame.shape[0]) * font_scale
        thickness = math.ceil(min(frame.shape[1], frame.shape[0]) * thickness_scale)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(frame, f'ID: {id}', (x1, round((y2 + y1) / 2)), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 255, 0),
                    thickness)

    return frame


def update_box_position(translation, box):
    """
        Updates the position of the tree with the estimated camera motion.
    """
    x1, y1, x2, y2, id, lifecounter, wasDetected = box
    box[0] = int(x1 + translation[0][0])
    box[2] = int(x2 + translation[0][0])

    if not wasDetected:
        box[5] += 1
    box[6] = False
    return box


def estimate_motion(prev_image, now_image, translation_history, direction_history):
    """
           Calculate the camera motion between two frames
    """

    prev_image_gray = cv2.cvtColor(prev_image, cv2.COLOR_BGR2GRAY)
    now_image_gray = cv2.cvtColor(now_image, cv2.COLOR_BGR2GRAY)

    keypoints1, descriptors1 = sift.detectAndCompute(prev_image_gray, None)
    keypoints2, descriptors2 = sift.detectAndCompute(now_image_gray, None)

    bf = cv2.BFMatcher(normType=NORM_HAMMING)
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.80 * n.distance]

    if len(good_matches) < 4:
        direction = np.mean(direction_history, axis=0)
        if direction[0][0] < 0:
            translation_vector = [[-200.0]]
        else:
            translation_vector = [[200.0]]
    else:
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        translation_vector = np.mean(dst_pts - src_pts, axis=0)

        if np.isnan(translation_vector).any():
            translation_vector = np.mean(translation_history, axis=0)
        else:
            direction_history.append(translation_vector)
            direction = np.mean(direction_history, axis=0)
            if direction[0][0] > 0:
                translation_vector = [[abs(translation_vector[0][0]), abs(translation_vector[0][1])]]
            else:
                translation_vector = [[-abs(translation_vector[0][0]), -abs(translation_vector[0][1])]]

        translation_history.append(translation_vector)

    if len(translation_history) > history_size:
        translation_history = translation_history[-history_size:]

    if len(direction_history) > direction_size:
        direction_history = direction_history[-direction_size:]

    if abs(translation_vector[0][0]) < 3:
        return [[0, 0]], translation_history, direction_history

    difference = abs(translation_vector[0][0]) - abs(np.mean(translation_history, axis=0)[0][0])

    if 0 < difference < 30:
        return np.mean(translation_history, axis=0), translation_history, direction_history
    else:
        if 0 < abs(translation_vector[0][0]) < 100 and (100 - abs(translation_vector[0][0])) < 40:
            translation_vector[0][0] += (translation_vector[0][0] * 75 / 100)
        elif abs(translation_vector[0][0]) < abs(np.mean(translation_history, axis=0)[0][0]):
            return np.mean(translation_history, axis=0), translation_history, direction_history

    return translation_vector, translation_history, direction_history


def del_box(box, now_image):
    """
       Check if the tree hasn't been detected for longer than the 'border_life' threshold and is outside the image.
         """
    x1, y1, x2, y2, id, lifecounter, wasDetected = box
    (height, width) = now_image.shape[:2]
    return (x1 < 0 or x2 > width) and lifecounter > border_life


def cut_out_tree(key, trees, file_name, folder_name, output_dir="output"):
    """
    Crop the image by the tree (identified by the key).
      """

    x1T, y1T, x2T, y2T, midT, currentFrame, oldTracker = trees[key]

    xstart, ystart, xstop, ystop = x1T / 2, y1T / 2, (640 + x2T) / 2, (640 + y2T) / 2

    if len(oldTracker) > 0:
        lowestx = 0
        highestx = currentFrame.shape[1]

        for oldTrackerKey in oldTracker:
            x1, y1, x2, y2, _, _, _ = oldTracker[oldTrackerKey]

            if x1T > x2 > 0:
                lowestx = x2
            if currentFrame.shape[1] > x1 > x2T:
                highestx = x1

        xstart = (x1T + lowestx) / 2
        xstop = (highestx + x2T) / 2

    scaling_factor_width = currentFrame.shape[1] / 640
    scaling_factor_height = currentFrame.shape[0] / 640

    xstart = int(xstart * scaling_factor_width)
    ystart = int(ystart * scaling_factor_height)
    xstop = int(xstop * scaling_factor_width)
    ystop = int(ystop * scaling_factor_height)

    crop = currentFrame[ystart:ystop, xstart:xstop]

    output_filename = f"{output_dir}/{folder_name}/{file_name}-Tree{key}.png"
    cv2.imwrite(output_filename, crop)

    del trees[key]

    return trees


def tracker_update_motion(tracker, translation, trees, frame, file_name, folder_name):
    """
    Update the positions of the tracked trees in the tracker using image-based estimated camera motion. Additionally,
    remove trees from the tracker that are no longer being tracked.
     """
    for key in list(tracker):
        tracker[key] = update_box_position(translation, tracker[key])

        if del_box(tracker[key], frame):
            trees = cut_out_tree(key, trees, file_name, folder_name)
            del tracker[key]

    return tracker, trees


def get_iou(box, tracker_box):
    """
    Calculate the Intersection over Union (IoU) between two bounding boxes
    """
    x_left = max(box[0], tracker_box[0])
    y_top = max(box[1], tracker_box[1])
    x_right = min(box[2], tracker_box[2])
    y_bottom = min(box[3], tracker_box[3])

    box1_area = (box[2] - box[0]) * (box[3] - box[1])
    box2_area = (tracker_box[2] - tracker_box[0]) * (tracker_box[3] - tracker_box[1])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    union_area = box1_area + box2_area - intersection_area

    iou = intersection_area / union_area
    return float(iou)


def combine_box_with_track_iou(tracker_tree_box, track_id, tracker, detection):
    """
    Search for the tracked tree (from the tracker) that has the biggest iou to a specific tree (tracker_tree_box) and
    update the position of the tracked tree in the tracker accordingly.
    """
    detection_id = 0
    biggest_iou = 0
    biggest_iou_compare = 0

    for box in detection:
        x1, y1, x2, y2, confidence, label = detection[box]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        iou = get_iou([x1, y1, x2, y2], tracker_tree_box)

        if iou == 0:
            continue

        if iou > biggest_iou:
            biggest_iou = iou
            detection_id = box
            x1final, y1final, x2final, y2final = x1, y1, x2, y2

    compareID = 0
    compareLife = 99999999

    if biggest_iou > 0.1:
        for key in tracker:
            if tracker[key][6] == True:
                continue

            compareiou = get_iou([x1final, y1final, x2final, y2final], tracker[key])

            if compareiou == 0:
                continue

            if compareiou > biggest_iou_compare and tracker[key][5] <= compareLife:
                biggest_iou_compare = compareiou
                compareID = key
                compareLife = tracker[key][5]

    if biggest_iou > 0.1 and track_id == compareID:
        tracker[track_id] = [x1final, y1final, x2final, y2final, track_id, 0, True]
        del detection[detection_id]

    return tracker, detection


def combine_box_with_track_distance(tree_key, tracker, id_generator, distance_arr, remaining_detections):
    """
    Search for the tracked tree (from the tracker) that has the smallest distance to a specific tree (with the
    tree_key) and update the position of the tracked tree in the tracker accordingly.
    """
    tree_box = remaining_detections[tree_key]
    smallest_distance = 9999999
    selected_tracker_id = id_generator

    tree_x1, tree_y1, tree_x2, tree_y2, _, _ = map(int, tree_box)
    direction = np.mean(direction_history, axis=0)

    for tracker_id, tracker_box in tracker.items():
        if tracker_box[6]:  # If the tracker box is marked for deletion, skip it
            continue

        tracker_x1, _, tracker_x2, _ = map(int, tracker_box[:4])

        center_tree = (tree_x1 + tree_x2) // 2
        center_tracker_tree = (tracker_x1 + tracker_x2) // 2
        new_distance = abs(center_tree - center_tracker_tree)

        if (
                (new_distance <= 0 and smallest_distance <= 0 and new_distance > smallest_distance) or
                (new_distance >= 0 and smallest_distance >= 0 and new_distance < smallest_distance) or
                (new_distance <= 0 and smallest_distance >= 0 and new_distance < smallest_distance)
        ):
            if tracker_x1 < 0 and (640 - tree_x2) < 50:
                continue
            if tracker_x2 > 640 and (0 - tree_x1) > -50:
                continue

            smallest = True

            for compare_tree_key, compare_tree_box in remaining_detections.items():
                if compare_tree_key == tree_key:
                    continue

                center_compare_tree = (
                        (int(compare_tree_box[0]) + int(compare_tree_box[2])) // 2
                )
                compare_distance = abs(center_compare_tree - center_tracker_tree)

                if (
                        (0 > new_distance >= compare_distance and compare_distance < 0) or
                        (0 < new_distance <= compare_distance and compare_distance > 0) or
                        (new_distance < 0 < compare_distance)
                ):
                    continue

                smallest = False
                break

            if smallest:
                smallest_distance = new_distance
                selected_tracker_id = tracker_id

    if smallest_distance != 9999999:
        if nowDistance:
            distance = lastDistance
            distance_out_of_frame = max(distance_arr)
        else:
            if not distance_arr:
                distance = 200
                distance_out_of_frame = 200
            else:
                distance_out_of_frame = max(distance_arr)
                distance = min(distance_arr)

        if (
                (direction[0][0] > 0 and tracker_x1 > 640 and (640 - tree_x2) <= 50) or
                (direction[0][0] < 0 and tracker_x2 < 0 and (0 + tree_x1) <= 50)
        ) and smallest_distance < all_time_biggest_distance:
            tracker[selected_tracker_id] = [
                tree_x1, tree_y1, tree_x2, tree_y2, selected_tracker_id, 0, True
            ]
        elif (
                (direction[0][0] > 0 and tracker_x1 > 640) or
                (direction[0][0] < 0 and tracker_x2 < 0)
        ) and smallest_distance > distance_out_of_frame:
            id_generator += 1
            tracker[id_generator] = [
                tree_x1, tree_y1, tree_x2, tree_y2, id_generator, 0, True
            ]
        elif smallest_distance > distance or id_generator == 0:
            id_generator += 1
            tracker[id_generator] = [
                tree_x1, tree_y1, tree_x2, tree_y2, id_generator, 0, True
            ]
        else:
            tracker[selected_tracker_id] = [
                tree_x1, tree_y1, tree_x2, tree_y2, selected_tracker_id, 0, True
            ]
    else:
        id_generator += 1
        tracker[id_generator] = [
            tree_x1, tree_y1, tree_x2, tree_y2, id_generator, 0, True
        ]

    del remaining_detections[tree_key]
    return tracker, id_generator, remaining_detections


def update_tracker(tracker, detections, id_generator, distance_arr):
    """
       Updates tracker with new detections.
       First with IOU and the remaining detections with distance
       """
    detection_dic = {}

    if len(detections[0].boxes.data) > 0:
        for dic_id, box in enumerate(detections[0].boxes.data):
            detection_dic[dic_id] = box

        keys_to_remove = set()

        for key1 in detection_dic:
            for key2 in detection_dic:
                if key1 != key2:
                    box1 = detection_dic[key1]
                    box2 = detection_dic[key2]
                    x1, y1, x2, y2, _, _ = box1
                    x12, y12, x22, y22, _, _ = box2
                    center1 = (x1 + x2) // 2
                    center2 = (x12 + x22) // 2
                    if abs(center1 - center2) <= 100:
                        keys_to_remove.add(key1)

        for key in keys_to_remove:
            del detection_dic[key]

    direction = np.mean(direction_history, axis=0)

    if direction_history and direction[0][0] < 0:
        sorted_tuples = sorted(detection_dic.items(), key=lambda x: x[1][0])
    else:
        sorted_tuples = sorted(detection_dic.items(), key=lambda x: x[1][0], reverse=True)

    remaining_detections = {k: v for k, v in sorted_tuples}

    for key in list(tracker):
        tracker, remaining_detections = combine_box_with_track_iou(tracker[key], key, tracker, remaining_detections)

    for box in list(remaining_detections):
        tracker, id_generator, remaining_detections = combine_box_with_track_distance(box, tracker, id_generator,
                                                                                      distance_arr,
                                                                                      remaining_detections)

    return tracker, id_generator


def update_trees(tracker, trees, frame):
    """
    Updates the trees list if the tracked tree is more central in the current image.

    Parameters:
    tracker (list): tracker list with tracked trees
    trees (list): trees list with trees and image where the tree is the most central in the image
    frame: current frame

    Returns:
    list: updated trees.

    """
    for key in tracker:
        if not tracker[key][6]:
            continue
        x1, y1, x2, y2 = tracker[key][:4]
        mid = (x2 + x1) / 2
        if key in trees:
            x1T, y1T, x2T, y2T, midT, currentFrame, oldTracker = trees[key]
            distanceNeu = max(320, mid) - min(320, mid)
            distanceAlt = max(320, midT) - min(320, midT)
            if distanceNeu < distanceAlt:
                copy = tracker.copy()
                del copy[key]
                trees[key] = [x1, y1, x2, y2, mid, frame, copy]
                continue
        else:
            copy = tracker.copy()
            del copy[key]
            trees[key] = [x1, y1, x2, y2, mid, frame, copy]

    return trees


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        for key in tracker:
            cut_out_tree(key, trees, file_name, folder_name)
        break

    now_image = cv2.resize(frame.copy(), (int(640), int(640)))
    original = frame.copy()

    results = model.predict(now_image, conf=0.40, iou=0.0)
    if is_first_frame:
        if make_video:
            out = cv2.VideoWriter(f"output/{folder_name}.avi", cv2.VideoWriter_fourcc(*'DIVX'), fps,
                                  (original.shape[1], original.shape[0]))

        frame_new = show_results(tracker, original.copy())
        prev_image = now_image
        is_first_frame = False
        trees = update_trees(tracker, trees, original)

        if make_video:
            out.write(frame_new)
        continue

    nowDistance = False
    if len(results[0].boxes.data) >= 2:
        box1 = results[0].boxes.data[0]
        box2 = results[0].boxes.data[1]
        x1, y1, x2, y2, confidence, label = box1
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        x12, y12, x22, y22, confidence2, label2 = box2
        x12 = int(x12)
        y12 = int(y12)
        x22 = int(x22)
        y22 = int(y22)
        center1 = round((x2 + x1) / 2)
        center2 = round((x22 + x12) / 2)
        newdistance = max(center1, center2) - min(center1, center2)

        if newdistance > all_time_biggest_distance:
            all_time_biggest_distance = newdistance
        if newdistance < all_time_smallest_distance:
            all_time_smallest_distance = newdistance

        if len(distance_arr) < 4:
            distance_arr.append(newdistance)
        elif len(distance_arr) == 4:
            distance_arr.append(newdistance)
            distance_arr.sort()
            distance_arr.pop(-1)
        lastDistance = newdistance
        nowDistance = True

    translation, translation_history, direction_history = estimate_motion(prev_image, now_image,
                                                                          translation_history,
                                                                          direction_history)

    tracker, trees = tracker_update_motion(tracker, translation, trees, now_image, file_name, folder_name)

    tracker, id_generator = update_tracker(tracker, results, id_generator,
                                           distance_arr)

    trees = update_trees(tracker, trees, original)

    frame_new = show_results(tracker, original.copy())

    if make_video:
        out.write(frame_new)
    prev_image = now_image.copy()

cap.release()
cv2.destroyAllWindows()
