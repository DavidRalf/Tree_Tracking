import math
import os

import cv2
import numpy as np
import pyproj
from geographiclib.geodesic import Geodesic
from pyproj import Geod, geod
from scipy.stats import norm, multivariate_normal, vonmises
from shapely import Point
from sklearn.linear_model import LinearRegression
from shapely.geometry import LineString
from ultralytics import YOLO
from geopy.distance import geodesic
from cv2 import NORM_HAMMING


class Tracker:
    def __init__(self, yolo, folder_name, file_name, matrix, side):

        if not os.path.exists('output'):
            os.makedirs('output')

        if not os.path.exists(f'output/{folder_name}'):
            os.makedirs(f'output/{folder_name}')

        self.matrix = matrix
        self.folder_name = folder_name
        self.file_name = file_name
        self.direction_size = 50
        self.history_size = 10
        self.sift = cv2.ORB_create(500, edgeThreshold=0, nlevels=20)
        self.bf = cv2.BFMatcher(normType=NORM_HAMMING)
        self.yolo = yolo

        self.side = side
        self.id_generator = 0
        self.border_life = 25
        self.trees = {}
        self.now_image_resized = None
        self.tracker = {}
        self.tree_3d_positions = {}
        self.translation_vector = None
        self.translation_history = []

        self.is_first_frame = True
        self.gps = None
        self.direction_of_travel = None
        self.camera_angle_to_direction = None
        self.prev_image_resized = None
        self.original_now_image = None
        self.all_time_smallest_distance = 999999
        self.all_time_biggest_distance = -1
        self.distance_arr = []
        self.nowDistance = None
        self.lastDistance = None
        self.direction_history = []

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
            # Berechnung der Differenz zwischen den GPS-Standorten
            # differenz = np.array(self.gps) - np.array(gps)

            # Berechnung der Fahrtrichtung in Grad
            # self.direction_of_travel = np.degrees(np.arctan2(differenz[1], differenz[0]))
            geod = Geodesic.WGS84
            result = geod.Inverse(gps[0], gps[1], self.gps[0], self.gps[1])
            initial_bearing = result['azi1']
            self.direction_of_travel = initial_bearing
            print(f"self.direction_of_travel {self.direction_of_travel}")

            # Der berechnete Winkel kann negative Werte haben, deshalb:
            # if self.direction_of_travel < 0:
            #    self.direction_of_travel += 360  # Umrechnung in den Bereich von 0 bis 360 Grad

            if self.side == "Left":
                self.camera_angle_to_direction = (self.direction_of_travel + 90) % 360
            else:
                self.camera_angle_to_direction = (self.direction_of_travel - 90) % 360
            self.gps = gps

    def update_image(self, image):
        self.original_now_image = image
        now_image_resized = cv2.resize(image, (int(640), int(640)))
        self.now_image_resized = now_image_resized
        results = self.yolo.predict(now_image_resized, conf=0.50, iou=0.0)
        print(results[0].boxes.data)
        results = list(results[0].boxes.data)
        print(results)
        direction = np.mean(self.direction_history, axis=0)

        if self.direction_history and direction[0][0] < 0:
            # Sort the list based on the first element of the tensor within the list
            results.sort(key=lambda x: x[0].item())
        else:
            results.sort(key=lambda x: x[0].item(), reverse=True)

        if self.is_first_frame:
            self.is_first_frame = False
            self.prev_image_resized = now_image_resized
        else:
            results = self.check_results(results)
            self.nowDistance = False
            self.calculate_pixel_distance(results)
            self.estimate_motion(now_image_resized)
            self.tracker_update_motion()
            self.update_tracker(results)
            self.update_trees()
            print(f"self.tree_3d_positions {self.tree_3d_positions}")
            self.prev_image_resized = now_image_resized
            for i in self.tree_3d_positions:
                print(f"lat {self.tree_3d_positions[i][0]} lon {self.tree_3d_positions[i][1]} key {i}")

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
            self.nowDistance = True

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
            translation_vector[0][0] += (translation_vector[0][0] * 20 / 100)
            self.translation_history.append(translation_vector)

        if len(self.translation_history) > self.history_size:
            self.translation_history = self.translation_history[-self.history_size:]

        if len(self.direction_history) > self.direction_size:
            self.direction_history = self.direction_history[-self.direction_size:]
        if not np.isnan(mean).any():
            difference = max(abs(translation_vector[0][0]), abs(mean[0][0])) - min(
                abs(translation_vector[0][0]), abs(mean[0][0]))

            if abs(translation_vector[0][0]) < 3 and difference < 10:
                self.translation_vector = [[0, 0]]
                pass

            if abs(translation_vector[0][0]) < abs(mean[0][0]):
                self.translation_vector = mean
                pass
            else:
                if 0 < abs(translation_vector[0][0]) < 100 and (100 - abs(translation_vector[0][0])) < 40:
                    translation_vector[0][0] += (translation_vector[0][0] * 75 / 100)
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
        print(f"{output_dir}/{self.folder_name}/{self.file_name}-Tree{key}.png")
        output_filename = f"{output_dir}/{self.folder_name}/{self.file_name}-Tree{key}.png"
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

        tree_x1, tree_y1, tree_x2, tree_y2, _, _ = map(int, tree_box)
        direction = np.mean(self.direction_history, axis=0)

        for tracker_id, tracker_box in self.tracker.items():
            if tracker_box[6]:  # If the tracker box is already matched, skip it
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
            if self.nowDistance:
                distance = self.lastDistance
                distance_out_of_frame = max(self.distance_arr)
            else:
                if not self.distance_arr:
                    distance = 200
                    distance_out_of_frame = 200
                else:
                    distance_out_of_frame = max(self.distance_arr)
                    distance = min(self.distance_arr)

            if (
                    (direction[0][0] > 0 and tracker_x1 > 640 and (640 - tree_x2) <= 50) or
                    (direction[0][0] < 0 and tracker_x2 < 0 and (0 + tree_x1) <= 50)
            ) and smallest_distance < self.all_time_biggest_distance:
                self.tracker[selected_tracker_id] = [
                    tree_x1, tree_y1, tree_x2, tree_y2, selected_tracker_id, 0, True
                ]
            elif (
                    (direction[0][0] > 0 and tracker_x1 > 640) or
                    (direction[0][0] < 0 and tracker_x2 < 0)
            ) and smallest_distance > distance_out_of_frame:
                self.id_generator += 1
                self.tracker[self.id_generator] = [
                    tree_x1, tree_y1, tree_x2, tree_y2, self.id_generator, 0, True
                ]
            elif smallest_distance > distance or self.id_generator == 0:
                self.id_generator += 1
                self.tracker[self.id_generator] = [
                    tree_x1, tree_y1, tree_x2, tree_y2, self.id_generator, 0, True
                ]
            else:
                self.tracker[selected_tracker_id] = [
                    tree_x1, tree_y1, tree_x2, tree_y2, selected_tracker_id, 0, True
                ]
        else:
            self.id_generator += 1
            self.tracker[self.id_generator] = [
                tree_x1, tree_y1, tree_x2, tree_y2, self.id_generator, 0, True
            ]

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
            # self.tree_3d_positions[key] = self.get_position(x1, y1, x2, y2)
            mid = (x2 + x1) / 2
            if key in self.trees:
                x1T, y1T, x2T, y2T, midT, currentFrame, oldTracker = self.trees[key]
                distanceNeu = max(320, mid) - min(320, mid)
                distanceAlt = max(320, midT) - min(320, midT)

                if distanceNeu < distanceAlt:
                    copy = self.tracker.copy()
                    del copy[key]
                    self.tree_3d_positions[key] = self.get_position(x1, y1, x2, y2)

                    self.trees[key] = [x1, y1, x2, y2, mid, self.original_now_image, copy]
                    continue
            else:
                copy = self.tracker.copy()
                del copy[key]

                self.tree_3d_positions[key] = self.get_position(x1, y1, x2, y2)

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
        self.prev_image_resized = None
        self.original_now_image = None
        self.all_time_smallest_distance = 999999
        self.all_time_biggest_distance = -1
        self.distance_arr = []
        self.nowDistance = None
        self.lastDistance = None
        self.direction_history = []
        self.tree_3d_positions = {}
        self.direction_of_travel = None
        self.camera_angle_to_direction = None

    def check_results(self, results):
        # Vorhersage des neues GPS punkts besser macen
        if len(self.tree_3d_positions) > 3:
            filtered_results = []
            check = [None]
            if len(self.tree_3d_positions) == 4:
                check = [None, -1, -2]
            if len(self.tree_3d_positions) == 5:
                check = [None, -1, -2, -3]
            if len(self.tree_3d_positions) >= 6:
                check = [None, -1, -2, -3, -4]
            for index, box in enumerate(results):
                finished = None
                x1, y1, x2, y2, confidence, label = box

                new_position = self.get_position(x1, y1, x2, y2)
                probability_at_new_position = -1

                total_distance = 0
                keys = list(self.tree_3d_positions.keys())
                # Calculate total distance between consecutive positions
                counter = 0
                distances = []
                for i in keys:
                    if i + 1 in self.tree_3d_positions:  # Check if the next position exists in the dictionary
                        coord1 = self.tree_3d_positions[i]
                        coord2 = self.tree_3d_positions[i + 1]
                        # distance = (coord2 - coord1)
                        geod = Geodesic.WGS84
                        result = geod.Inverse(coord1[0], coord1[1], coord2[0], coord2[1])
                        bearing = result['azi1']
                        distance = result['s12']
                        print(f"distance {distance}")
                        distances.append(distance)
                        total_distance += distance
                        counter += 1

                # Calculate average distance
                # average_distance = total_distance / counter
                average_distance = np.mean(distances, axis=0)
                print(f"total_distance {total_distance}")
                print(f"average_distance meter {average_distance}")
                print(f"average_distance degree {average_distance / 111000}")

                key = list(self.tree_3d_positions.keys())[-1]

                # Calculate the geodesic distance between the points

                variance_x = 0.000001175975741982

                # variance_x = np.sqrt(np.var(distances,axis=0))
                print(f"variance {variance_x}")
                print(f"variance {variance_x * 111000}")
                print(f"variance_x2 {np.sqrt(variance_x) * 111000}")
                print(f"np.var(distances) {np.var(distances)}")
                print(f"np.var(distances) degree {np.var(distances) / 111000}")
                print(f"np.var(distances) degree  sqrt{np.var(distances) / 111000}")
                print(f"np.var sqrt(distances) {np.sqrt(np.var(distances)) / 111000}")
                latitude_data = [self.tree_3d_positions[key][0] for key in self.tree_3d_positions]
                longitude_data = [self.tree_3d_positions[key][1] for key in self.tree_3d_positions]
                # Calculate sample variances
                variance_latitude = np.var(latitude_data)
                variance_longitude = np.var(longitude_data)
                print(f"variance_latitude {variance_latitude}")
                print(f"variance_longitude {variance_longitude}")
                print(f"variance_latitude {np.sqrt(variance_latitude)}")
                print(f"variance_longitude {np.sqrt(variance_longitude)}")
                variance_x = np.sqrt(np.var(distances)) / 111000
                variance_x = np.sqrt(variance_latitude)
                variance_y = np.sqrt(variance_longitude)
                variance_x = 0.000001175975741982

                variance_y = 0.000001175975741982

                print(f"self.tree_3d_positions[key][0 {self.tree_3d_positions[key][0]}")
                next_x = self.tree_3d_positions[key][0] + (average_distance)
                next_y = self.tree_3d_positions[key][1] + (average_distance)
                print(f"next_x {next_x}")
                print(f"next_y {next_y}")

                # Calculate the destination point using the bearing and distance
                direct_result = geod.Direct(self.tree_3d_positions[key][0], self.tree_3d_positions[key][1], bearing,
                                            average_distance)
                next_x = direct_result['lat2']
                next_y = direct_result['lon2']
                print(f"next_x {next_x}")
                print(f"next_y {next_y}")
                # Erstelle eine Gaussverteilung um die vorhergesagte Position des nächsten Baums
                # gauss_distribution_x = norm(loc=(next_x,next_y), scale=variance_x)

                # Calculate probability of 'YOLO' point within the normal distribution
                probability_latitude = norm.pdf(new_position[0], next_x, variance_x) / norm.pdf(next_x, next_x,
                                                                                                variance_x)
                probability_longitude = norm.pdf(new_position[1], next_y, variance_y) / norm.pdf(next_y, next_y,
                                                                                                 variance_y)
                print(f"probability_latitude {probability_latitude}")
                print(f"probability_longitude {probability_longitude}")
                # probability = gauss_distribution_x.pdf(new_position) / gauss_distribution_x.pdf((next_x,next_y))

                probability = (probability_latitude + probability_longitude) / 2
                print(f"probability {probability}")
                # probability_at_new_position_temp = np.mean(probability)
                probability_at_new_position_temp = probability

                if index + 1 < len(results):
                    x1_compare, y1_compare, x2_compare, y2_compare, confidence_compare, label_compare = results[
                        index + 1]
                    new_position_compare = self.get_position(x1_compare, y1_compare, x2_compare, y2_compare)
                    # probability_compare = gauss_distribution_x.pdf(new_position_compare) / gauss_distribution_x.pdf(
                    #    (next_x,next_y))
                    probability_latitude_compare = norm.pdf(new_position_compare[0], next_x, variance_x) / norm.pdf(
                        next_x, next_x,
                        variance_x)
                    probability_longitude_compare = norm.pdf(new_position_compare[1], next_y, variance_y) / norm.pdf(
                        next_y, next_y,
                        variance_y)
                    probability_compare = (probability_latitude_compare + probability_longitude_compare) / 2

                    probability_compare = np.mean(probability_compare)
                    if probability_at_new_position_temp > probability_compare and probability_at_new_position_temp > probability_at_new_position:
                        probability_at_new_position = probability_at_new_position_temp
                elif probability_at_new_position_temp > probability_at_new_position:
                    probability_at_new_position = probability_at_new_position_temp

                print(f"for check {check}")
                for checking in check:
                    print(f"checking {checking}")
                    if checking is None:
                        continue
                    key = list(self.tree_3d_positions.keys())[checking]
                    position = self.tree_3d_positions[key]
                    print(f"key {key}")
                    print(f"position {position}")
                    print(f"position {position[0]}")
                    # gauss_distribution_x = norm(loc=position, scale=variance_x)

                    # probability = gauss_distribution_x.pdf(new_position) / gauss_distribution_x.pdf(position)

                    probability_latitude = norm.pdf(new_position[0], position[0], variance_x) / norm.pdf(position[0],
                                                                                                         position[0],
                                                                                                         variance_x)
                    probability_longitude = norm.pdf(new_position[1], position[1], variance_y) / norm.pdf(position[1],
                                                                                                          position[1],
                                                                                                          variance_y)
                    print(f"probability_latitude {probability_latitude}")
                    probability = (probability_latitude + probability_longitude) / 2
                    print(f"probability {probability}")

                    probability_at_new_position_temp = probability

                    if probability_at_new_position_temp >= probability_at_new_position:
                        probability_at_new_position = probability_at_new_position_temp
                        finished = checking

                if ((confidence + probability_at_new_position) / 2) >= 0.4:
                    filtered_results.append(box)
                    print(f"finished {finished}")
                    print(f"check {check}")
                    if finished in check:
                        index_to_remove = check.index(finished)
                        del check[index_to_remove:]  # Remove elements after 'finished'

            results = filtered_results
        return results

    def get_position(self, x1, y1, x2, y2):
        scaling_factor_width = self.original_now_image.shape[1] / 640
        scaling_factor_height = self.original_now_image.shape[0] / 640
        x1 = int(x1 * scaling_factor_width)
        y1 = int(y1 * scaling_factor_height)
        x2 = int(x2 * scaling_factor_width)
        y2 = int(y2 * scaling_factor_height)
        center_x = round((x2 + x1) / 2)
        center_y = round((y2 + y1) / 2)

        print(f"center_x {center_x}")
        print(f"center_y {center_y}")

        p = np.array([center_x, center_y, 1])
        K_inv = np.linalg.inv(self.matrix)
        c = 1 * np.dot(K_inv, p)
        print(f"c {c}")
        # Bestimmung der Rotationswinkel, um die Kamera nach Norden auszurichten
        print(f"rotation_angle_to_north vorher {self.camera_angle_to_direction}")
        rotation_angle_to_north = (0 - self.camera_angle_to_direction)
        print(f"rotation_angle_to_north nacher {rotation_angle_to_north}")

        # Konvertierung des Rotationswinkels in Radiant für die Rotation
        rotation_angle_to_north = np.radians(rotation_angle_to_north)
        print(f"rotation_angle_to_north radian {rotation_angle_to_north}")

        # Aufbau der Rotationsmatrix für die zusätzliche Rotation, um die Kamera nach Norden auszurichten
        rotation_matrix_to_north = np.array(
            [[np.cos(rotation_angle_to_north), -np.sin(rotation_angle_to_north)],
             [np.sin(rotation_angle_to_north), np.cos(rotation_angle_to_north)]])
        print(f"rotation_matrix_to_north  {rotation_matrix_to_north}")
        # Baumposition relativ zur Kamera (nur x und y) für die Rotation

        tree_position_camera_2d = [
            c[0],
            c[2]
        ]

        # Umrechnung der Baumposition relativ zur Kamera von Metern in Grad

        # Transformation der Baumposition relativ zur Kamera in die Ausrichtung nach Norden

        tree_position_camera_to_north = np.dot(rotation_matrix_to_north, tree_position_camera_2d)
        print(f"tree_position_camera_to_north  {tree_position_camera_to_north}")

        tree_position_camera_to_north_degree = tree_position_camera_to_north / 111000
        print(f"tree_position_camera_to_north_degree  {tree_position_camera_to_north_degree}")

        # Hinzufügen der umgerechneten relativen Baumposition in Grad zur aktuellen GPS-Position

        print(f"self.gps {self.gps}")
        tree_position_global = self.gps + tree_position_camera_to_north_degree

        print(f"tree_position_global_longitude {tree_position_global[1]}")
        print(f"tree_position_global_latitude {tree_position_global[0]}")

        return tree_position_global
