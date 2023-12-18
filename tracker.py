import math
import os

import cv2
import numpy as np
from geographiclib.geodesic import Geodesic
from scipy.stats import norm
from shapely.geometry import Point
from cv2 import NORM_HAMMING
import geojson


class Tracker:
    def __init__(self, yolo, folder_name, file_name, matrix, side, gps_check):

        if not os.path.exists('output'):
            os.makedirs('output')

        if not os.path.exists(f'output/{folder_name}'):
            os.makedirs(f'output/{folder_name}')

        self.row_line = None
        self.gps_check = gps_check
        self.matrix = matrix
        self.folder_name = folder_name
        self.file_name = file_name
        self.direction_size = 50
        self.history_size = 10
        self.sift = cv2.ORB_create(500, edgeThreshold=0, nlevels=20)
        self.bf = cv2.BFMatcher(normType=NORM_HAMMING)
        self.yolo = yolo
        self.testing_array = []
        self.side = side
        self.id_generator = 0
        self.border_life = 25
        self.trees = {}
        self.now_image_resized = None
        self.tracker = {}
        self.tree_positions_for_predicting = {}
        self.tree_gps_positions = {}
        self.translation_vector = None
        self.translation_history = []

        self.is_first_frame = True
        self.gps = None
        self.camera_angle = None
        self.direction_of_travel_for_predicting = None
        self.camera_angle_to_direction = None
        self.prev_image_resized = None
        self.original_now_image = None
        self.all_time_smallest_distance = 999999
        self.all_time_biggest_distance = -1
        self.distance_arr = []
        self.lastDistance = None
        self.direction_history = []
        self.row = None

    def set_row_line(self, row_line):
        self.row_line=row_line
    def set_row(self, row):
        self.row = row
        if not os.path.exists(f'output/{self.folder_name}/{row}'):
            os.makedirs(f'output/{self.folder_name}/{row}')
            os.makedirs(f'output/{self.folder_name}/{row}/images')

    def make_geojson(self):
        positions = self.get_gps()
        print(positions)
        coordinates = []
        for i in positions:
            coordinates.append(positions[i])

        point_features = [geojson.Feature(geometry=geojson.Point((coord[1], coord[0])), properties={}) for coord in
                          coordinates]

        # Create a GeoJSON Feature Collection
        feature_collection = geojson.FeatureCollection(point_features)

        # Write the GeoJSON Feature Collection to a file
        with open(f'output/{self.folder_name}/{self.row}/output.geojson', 'w') as f:
            geojson.dump(feature_collection, f)

    def get_gps(self):
        return self.tree_gps_positions

    def set_folder_name(self, name):
        self.folder_name = name
        if not os.path.exists(f'output/{name}'):
            os.makedirs(f'output/{name}')

    def set_file_name(self, name):
        self.file_name = name

    def update_gps(self, gps):
        if self.gps is None:
            self.gps = gps
        else:
            geod = Geodesic.WGS84
            result = geod.Inverse(gps[0], gps[1], self.gps[0], self.gps[1])
            initial_bearing = result['azi1']
            self.direction_of_travel_for_predicting = initial_bearing % 360

            result = geod.Inverse(self.gps[0], self.gps[1], gps[0], gps[1])
            initial_bearing = result['azi1'] % 360

            if self.side == "Left":
                self.camera_angle_to_direction = (self.direction_of_travel_for_predicting + 90) % 360
                self.camera_angle = (initial_bearing + 90) % 360
            else:
                self.camera_angle_to_direction = (self.direction_of_travel_for_predicting - 90) % 360
                self.camera_angle = (initial_bearing - 90) % 360
            self.gps = gps

    def update_image(self, image):
        self.original_now_image = image
        now_image_resized = cv2.resize(image, (int(640), int(640)))
        self.now_image_resized = now_image_resized
        results = self.yolo.predict(now_image_resized, conf=0.50, iou=0.0)
        results = list(results[0].boxes.data)
        direction = np.mean(self.direction_history, axis=0)

        if self.direction_history and direction[0][0] < 0:
            results.sort(key=lambda x: x[0].item())
        else:
            results.sort(key=lambda x: x[0].item(), reverse=True)

        if self.is_first_frame:
            self.is_first_frame = False
            self.prev_image_resized = now_image_resized
        else:
            self.calculate_pixel_distance(results)
            self.estimate_motion(now_image_resized)
            self.tracker_update_motion()
            self.update_tracker(results)
            self.update_trees()
            self.prev_image_resized = now_image_resized

    def calculate_pixel_distance(self, results):
        if len(results) >= 2:
            box1 = results[0]
            box2 = results[1]
            x1, y1, x2, y2, confidence, label = box1
            x1 = int(x1)
            x2 = int(x2)
            x12, y12, x22, y22, confidence2, label2 = box2
            x12 = int(x12)
            x22 = int(x22)
            center1 = round((x2 + x1) / 2)
            center2 = round((x22 + x12) / 2)
            new_distance = max(center1, center2) - min(center1, center2)

            if new_distance > self.all_time_biggest_distance:
                self.all_time_biggest_distance = new_distance
            if new_distance < self.all_time_smallest_distance:
                self.all_time_smallest_distance = new_distance

            if len(self.distance_arr) < 4:
                self.distance_arr.append(new_distance)
            elif len(self.distance_arr) == 4:
                self.distance_arr.append(new_distance)
                self.distance_arr.sort()
                self.distance_arr.pop(-1)
            self.lastDistance = new_distance

    def estimate_motion(self, now_image_resized):
        """
        Calculate the camera motion between two frames
        """
        prev_image_gray = cv2.cvtColor(self.prev_image_resized, cv2.COLOR_BGR2GRAY)
        now_image_gray = cv2.cvtColor(now_image_resized, cv2.COLOR_BGR2GRAY)

        keypoints1, descriptors1 = self.sift.detectAndCompute(prev_image_gray, None)
        keypoints2, descriptors2 = self.sift.detectAndCompute(now_image_gray, None)

        matches = self.bf.knnMatch(descriptors1, descriptors2, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.80 * n.distance]

        if len(good_matches) < 4:
            direction = np.mean(self.direction_history, axis=0)
            if direction[0][0] < 0:
                translation_vector = [[-200.0]]
            else:
                translation_vector = [[200.0]]
        else:
            src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            translation_vector = np.mean(dst_pts - src_pts, axis=0)
            mean = np.mean(self.translation_history, axis=0)

            if np.isnan(translation_vector).any():
                translation_vector = np.mean(self.translation_history, axis=0)
            else:
                self.direction_history.append(translation_vector)
                direction = np.mean(self.direction_history, axis=0)
                if direction[0][0] > 0:
                    translation_vector = [[abs(translation_vector[0][0]), abs(translation_vector[0][1])]]
                else:
                    translation_vector = [[-abs(translation_vector[0][0]), -abs(translation_vector[0][1])]]
            self.translation_history.append(translation_vector)
            # translation_vector[0][0] += (translation_vector[0][0] * 20 / 100)

        if len(self.translation_history) > self.history_size:
            self.translation_history = self.translation_history[-self.history_size:]

        if len(self.direction_history) > self.direction_size:
            self.direction_history = self.direction_history[-self.direction_size:]
        if not np.isnan(mean).any():
            print(f"translation_vector {translation_vector}")
            difference = max(abs(translation_vector[0][0]), abs(mean[0][0])) - min(
                abs(translation_vector[0][0]), abs(mean[0][0]))
            print(f"mean {mean}")
            print(f"difference {difference}")
            if abs(translation_vector[0][0]) < 3 and difference < 10:
                self.translation_vector = [[0, 0]]
                return
            if abs(translation_vector[0][0]) < abs(mean[0][0]):
                self.translation_vector = mean
                return
            # else:
            #    if 0 < abs(translation_vector[0][0]) < 100 and (100 - abs(translation_vector[0][0])) < 40:
            #        translation_vector[0][0] += (translation_vector[0][0] * 25 / 100)

        self.translation_vector = translation_vector

    def tracker_update_motion(self):
        """
            Update the positions of the tracked trees in the tracker using image-based estimated camera motion. Additionally,
            remove trees from the tracker that are no longer being tracked.
        """
        for key in list(self.tracker):
            self.tracker[key] = self.update_box_position(self.translation_vector, self.tracker[key])

            if self.del_box(self.tracker[key], self.now_image_resized):
                self.cut_out_tree(key)
                del self.tracker[key]

    def update_box_position(self, translation, detected_tree):
        """
            Updates the position of the tree with the estimated camera motion.
        """
        x1, y1, x2, y2, id, lifecounter, wasDetected = detected_tree
        detected_tree[0] = int(x1 + translation[0][0])
        detected_tree[2] = int(x2 + translation[0][0])
        print(f"translation {translation}")
        if not wasDetected:
            detected_tree[5] += 1
        detected_tree[6] = False
        return detected_tree

    def del_box(self, detected_tree, now_image_resized):
        """
           Check if the tree hasn't been detected for longer than the 'border_life' threshold and is outside the image.
        """
        x1, y1, x2, y2, id, lifecounter, wasDetected = detected_tree
        (height, width) = now_image_resized.shape[:2]
        return (x1 < 0 or x2 > width) and lifecounter > self.border_life

    def cut_out_tree(self, key, output_dir="output"):
        """
            Crop the image by the tree (identified by the key).
        """

        x1T, y1T, x2T, y2T, midT, currentFrame, oldTracker = self.trees[key]

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
        output_filename = f"{output_dir}/{self.folder_name}/{self.row}/images/{self.file_name}-Tree{key}.png"
        cv2.imwrite(output_filename, crop)
        del self.trees[key]

    def update_tracker(self, detections):
        """
            Updates tracker with new detections.
            First with IOU and the remaining detections with distance
        """
        detection_dic = {}

        if len(detections) > 0:
            for dic_id, box in enumerate(detections):
                detection_dic[dic_id] = box
            keys_to_remove = set()
            for key1 in detection_dic:
                for key2 in detection_dic:
                    if key1 != key2:
                        box1 = detection_dic[key1]
                        box2 = detection_dic[key2]
                        x1, y1, x2, y2, _, _ = box1
                        x12, y12, x22, y22, _, _ = box2
                        center1 = (x1 + x2) / 2
                        center2 = (x12 + x22) / 2
                        if abs(center1 - center2) <= 100:
                            keys_to_remove.add(key1)

            for key in keys_to_remove:
                del detection_dic[key]

        direction = np.mean(self.direction_history, axis=0)

        if self.direction_history and direction[0][0] < 0:
            sorted_tuples = sorted(detection_dic.items(), key=lambda x: x[1][0])
        else:
            sorted_tuples = sorted(detection_dic.items(), key=lambda x: x[1][0], reverse=True)

        remaining_detections = {k: v for k, v in sorted_tuples}

        for key in list(self.tracker):
            remaining_detections = self.combine_box_with_track_iou(self.tracker[key], key, remaining_detections)

        for box in list(remaining_detections):
            remaining_detections = self.combine_box_with_track_distance(box, remaining_detections)

    def combine_box_with_track_iou(self, tracker_tree_box, track_id, detection):
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

            iou = self.get_iou([x1, y1, x2, y2], tracker_tree_box)

            if iou == 0:
                continue

            if iou > biggest_iou:
                biggest_iou = iou
                detection_id = box
                x1final, y1final, x2final, y2final = x1, y1, x2, y2

        compareID = 0
        compareLife = 99999999

        if biggest_iou > 0.1:
            for key in self.tracker:
                if self.tracker[key][6]:
                    continue

                compareiou = self.get_iou([x1final, y1final, x2final, y2final], self.tracker[key])

                if compareiou == 0:
                    continue

                if compareiou > biggest_iou_compare and self.tracker[key][5] <= compareLife:
                    biggest_iou_compare = compareiou
                    compareID = key
                    compareLife = self.tracker[key][5]

        if biggest_iou > 0.1 and track_id == compareID:
            self.tracker[track_id] = [x1final, y1final, x2final, y2final, track_id, 0, True]
            del detection[detection_id]

        return detection

    def combine_box_with_track_distance(self, tree_key, remaining_detections):
        """
            Search for the tracked tree (from the tracker) that has the smallest distance to a specific tree (with the
            tree_key) and update the position of the tracked tree in the tracker accordingly.
        """
        tree_box = remaining_detections[tree_key]
        smallest_distance = 9999999
        selected_tracker_id = self.id_generator

        tree_x1, tree_y1, tree_x2, tree_y2, tree_confidence, tree_label = tree_box
        tree_x1, tree_y1, tree_x2, tree_y2 = int(tree_x1), int(tree_y1), int(tree_x2), int(tree_y2)
        direction = np.mean(self.direction_history, axis=0)

        for tracker_id, tracker_box in self.tracker.items():
            if tracker_box[6]:  # If the tracker box is already matched, skip it
                continue

            tracker_x1, _, tracker_x2, _ = map(int, tracker_box[:4])

            center_tree = (tree_x1 + tree_x2) // 2
            center_tracker_tree = (tracker_x1 + tracker_x2) // 2
            new_distance = abs(center_tree - center_tracker_tree)

            if (
                    (0 >= new_distance > smallest_distance and smallest_distance <= 0) or
                    (0 <= new_distance < smallest_distance and smallest_distance >= 0) or
                    (new_distance <= 0 <= smallest_distance and new_distance < smallest_distance)
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

            if not self.distance_arr:
                distance = 200
                distance_out_of_frame = 200
            else:
                distance_out_of_frame = max(self.distance_arr)
                distance = min(self.distance_arr)
            if distance > 200:
                distance = 200
                distance_out_of_frame = 200

            if (
                    (direction[0][0] > 0 and tracker_x1 > 640 and (640 - tree_x2) <= 50) or
                    (direction[0][0] < 0 and tracker_x2 < 0 and (0 + tree_x1) <= 50)
            ) and smallest_distance < distance_out_of_frame:
                self.tracker[selected_tracker_id] = [
                    tree_x1, tree_y1, tree_x2, tree_y2, selected_tracker_id, 0, True
                ]
            elif (
                    (direction[0][0] > 0 and tracker_x1 > 640) or
                    (direction[0][0] < 0 and tracker_x2 < 0)
            ) and smallest_distance > distance_out_of_frame:
                self.check_results(tree_x1, tree_y1, tree_x2, tree_y2, tree_confidence)

            elif smallest_distance > distance or self.id_generator == 0:
                self.check_results(tree_x1, tree_y1, tree_x2, tree_y2, tree_confidence)

            else:
                self.tracker[selected_tracker_id] = [
                    tree_x1, tree_y1, tree_x2, tree_y2, selected_tracker_id, 0, True
                ]
        else:

            self.check_results(tree_x1, tree_y1, tree_x2, tree_y2, tree_confidence)

        del remaining_detections[tree_key]
        return remaining_detections

    def get_iou(self, box, tracker_box):
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

    def update_trees(self):
        """
            Updates the trees list if the tracked tree is more central in the current image.

            Parameters:
            tracker (list): tracker list with tracked trees
            trees (list): trees list with trees and image where the tree is the most central in the image
            frame: current frame

            Returns:
            list: updated trees.

        """
        for key in self.tracker:
            if not self.tracker[key][6]:
                continue
            print(f"key key {key}")
            x1, y1, x2, y2 = self.tracker[key][:4]
            mid = (x2 + x1) / 2
            if key in self.trees:
                x1T, y1T, x2T, y2T, midT, currentFrame, oldTracker = self.trees[key]
                distanceNeu = max(320, mid) - min(320, mid)
                distanceAlt = max(320, midT) - min(320, midT)

                if distanceNeu < distanceAlt:
                    copy = self.tracker.copy()
                    del copy[key]

                    self.tree_positions_for_predicting[key] = self.get_position_for_predicting(x1, y1, x2, y2)
                    self.tree_gps_positions[key] = self.get_position(x1, y1, x2, y2)
                    self.trees[key] = [x1, y1, x2, y2, mid, self.original_now_image, copy]
                    continue
            else:
                copy = self.tracker.copy()
                del copy[key]
                self.tree_positions_for_predicting[key] = self.get_position_for_predicting(x1, y1, x2, y2)
                self.tree_gps_positions[key] = self.get_position(x1, y1, x2, y2)
                self.trees[key] = [x1, y1, x2, y2, mid, self.original_now_image, copy]

    def show_results(self, image_size=(640, 640), font_scale=0.002, thickness_scale=0.002):
        """
            Add the bounding boxes to the frame for visualization.
        """
        frame = self.original_now_image.copy()
        for key, (x1, y1, x2, y2, id, life_counter, wasDetected) in self.tracker.items():
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
            cv2.putText(frame, f'ID: {id}', (x1, round((y2 + y1) / 2)), cv2.FONT_HERSHEY_SIMPLEX, font_size,
                        (0, 255, 0),
                        thickness)

        return frame

    def finish(self):
        for key in self.tracker:
            self.cut_out_tree(key)
        self.reset()

    def reset(self):
        self.id_generator = 0
        self.border_life = 25
        self.trees = {}
        self.now_image_resized = None
        self.tracker = {}
        self.translation_vector = None
        self.translation_history = []
        self.is_first_frame = True
        self.gps = None
        self.camera_angle = None
        self.prev_image_resized = None
        self.original_now_image = None
        self.all_time_smallest_distance = 999999
        self.all_time_biggest_distance = -1
        self.distance_arr = []
        self.lastDistance = None
        self.direction_history = []
        self.tree_gps_positions = {}
        self.tree_positions_for_predicting = {}
        self.direction_of_travel_for_predicting = None
        self.camera_angle_to_direction = None

    def check_results(self, tree_x1, tree_y1, tree_x2, tree_y2, tree_confidence):
        if self.gps_check:
            if len(self.tree_positions_for_predicting) > 3:

                new_position = self.get_position_for_predicting(tree_x1, tree_y1, tree_x2, tree_y2)

                keys = list(self.tree_positions_for_predicting.keys())

                distances = []
                for i in keys:
                    if i + 1 in self.tree_positions_for_predicting:
                        coord1 = self.tree_positions_for_predicting[i]
                        coord2 = self.tree_positions_for_predicting[i + 1]
                        geod = Geodesic.WGS84
                        result = geod.Inverse(coord1[0], coord1[1], coord2[0], coord2[1])
                        bearing = result['azi1']
                        distance = result['s12']
                        print(f"distance {distance}")
                        distances.append(distance)

                average_distance = np.mean(distances)

                key = list(self.tree_positions_for_predicting.keys())[-1]

                variance_x = 0.000001575975741982

                variance_y = 0.000001575975741982

                # Calculate the destination point using the bearing and distance
                direct_result = geod.Direct(self.tree_positions_for_predicting[key][0],
                                            self.tree_positions_for_predicting[key][1], bearing,
                                            average_distance)
                next_x = direct_result['lat2']
                next_y = direct_result['lon2']

                probability_latitude = norm.pdf(new_position[0], next_x, variance_x) / norm.pdf(next_x, next_x,
                                                                                                variance_x)
                probability_longitude = norm.pdf(new_position[1], next_y, variance_y) / norm.pdf(next_y, next_y,
                                                                                                 variance_y)

                probability = (probability_latitude + probability_longitude) / 2
                if ((tree_confidence + probability) / 2) >= 0.5:
                    self.id_generator += 1
                    self.tracker[self.id_generator] = [
                        tree_x1, tree_y1, tree_x2, tree_y2, self.id_generator, 0, True]
            else:
                self.id_generator += 1
                self.tracker[self.id_generator] = [
                    tree_x1, tree_y1, tree_x2, tree_y2, self.id_generator, 0, True]
        else:
            self.id_generator += 1
            self.tracker[self.id_generator] = [
                tree_x1, tree_y1, tree_x2, tree_y2, self.id_generator, 0, True]

    def get_position_for_predicting(self, x1, y1, x2, y2):
        scaling_factor_width = self.original_now_image.shape[1] / 640
        scaling_factor_height = self.original_now_image.shape[0] / 640
        x1 = int(x1 * scaling_factor_width)
        y1 = int(y1 * scaling_factor_height)
        x2 = int(x2 * scaling_factor_width)
        y2 = int(y2 * scaling_factor_height)
        center_x = round((x2 + x1) / 2)
        center_y = round((y2 + y1) / 2)

        depth = 1
        p = np.array([center_x, center_y, 1])
        K_inv = np.linalg.inv(self.matrix)
        c = depth * np.dot(K_inv, p)

        rotation_angle_to_north = (0 - self.camera_angle_to_direction)

        rotation_angle_to_north = np.radians(rotation_angle_to_north)

        rotation_matrix_to_north = np.array([
            [np.cos(rotation_angle_to_north), -np.sin(rotation_angle_to_north)],
            [np.sin(rotation_angle_to_north), np.cos(rotation_angle_to_north)]
        ])

        tree_position_camera_2d = [
            c[0],
            c[2]
        ]

        tree_position_camera_to_north = np.dot(rotation_matrix_to_north, tree_position_camera_2d)

        tree_position_camera_to_north_degree = tree_position_camera_to_north / 111000

        tree_position_global = self.gps + tree_position_camera_to_north_degree

        return tree_position_global

    def get_position(self, x1, y1, x2, y2):
        scaling_factor_width = self.original_now_image.shape[1] / 640
        scaling_factor_height = self.original_now_image.shape[0] / 640
        x1 = int(x1 * scaling_factor_width)
        y1 = int(y1 * scaling_factor_height)
        x2 = int(x2 * scaling_factor_width)
        y2 = int(y2 * scaling_factor_height)
        center_x = round((x2 + x1) / 2)
        center_y = round((y2 + y1) / 2)
        if self.row_line is None:
            depth = 1
        else:
            current_point = Point(self.gps[1], self.gps[0])
            distance_to_line = current_point.distance(self.row_line)
            depth = distance_to_line * 111000

        p = np.array([center_x, center_y, 1])
        K_inv = np.linalg.inv(self.matrix)
        c = depth * np.dot(K_inv, p)
        rotation_angle_to_north = (0 - self.camera_angle)

        rotation_angle_to_north = np.radians(rotation_angle_to_north)

        rotation_matrix_to_north = np.array([
            [np.cos(rotation_angle_to_north), -np.sin(rotation_angle_to_north)],
            [np.sin(rotation_angle_to_north), np.cos(rotation_angle_to_north)]
        ])

        tree_position_camera_2d = [
            c[0],
            c[2]
        ]

        tree_position_camera_to_north = np.dot(rotation_matrix_to_north, tree_position_camera_2d)
        tree_position_camera_to_north_degree = tree_position_camera_to_north / 111000

        tree_position_global = self.gps + tree_position_camera_to_north_degree

        return tree_position_global
