import os
import sys
import cv2
import geojson
import numpy as np
from ultralytics import YOLO
import rosbag
import pyzed.sl as sl
import synchronize as sync
from tracker import Tracker
from cv_bridge import CvBridge
from shapely.geometry import LineString, Point
from geographiclib.geodesic import Geodesic

camera_topics = {
    '2023-03-01_14-59-46_Horizontal 1.bag': ['/zedA/zed_node_A/rgb/image_rect_color',
                                             '/zedA/zed_node_A/right/image_rect_color',
                                             '/zedB/zed_node_B/rgb/image_rect_color',
                                             '/zedB/zed_node_B/right/image_rect_color'],
    '2023-03-01_15-21-21_30 Grad 1.bag': ['/zedA/zed_node_A/rgb/image_rect_color',
                                          '/zedA/zed_node_A/right/image_rect_color',
                                          '/zedB/zed_node_B/rgb/image_rect_color',
                                          '/zedB/zed_node_B/right/image_rect_color'],
    '2023-03-01_15-31-36_90 Grad 1.bag': ['/zedA/zed_node_A/rgb/image_rect_color',
                                          '/zedA/zed_node_A/right/image_rect_color',
                                          '/zedB/zed_node_B/rgb/image_rect_color',
                                          '/zedB/zed_node_B/right/image_rect_color'],
    '2023-03-01_16-00-43_90 Grad 2.bag': ['/zedA/zed_node_A/rgb/image_rect_color',
                                          '/zedA/zed_node_A/right/image_rect_color',
                                          '/zedB/zed_node_B/rgb/image_rect_color',
                                          '/zedB/zed_node_B/right/image_rect_color'],
    '2023-03-01_16-11-38_30 Grad 2.bag': ['/zedA/zed_node_A/rgb/image_rect_color',
                                          '/zedA/zed_node_A/right/image_rect_color',
                                          '/zedB/zed_node_B/rgb/image_rect_color',
                                          '/zedB/zed_node_B/right/image_rect_color'],
    '2023-03-01_16-34-04_Horizontal 2.bag': ['/zedA/zed_node_A/rgb/image_rect_color',
                                             '/zedA/zed_node_A/right/image_rect_color',
                                             '/zedB/zed_node_B/rgb/image_rect_color',
                                             '/zedB/zed_node_B/right/image_rect_color'],
    '2023-03-02_09-21-59_Reihe 2 langsam.bag': ['/zedA/zed_node_A/rgb/image_rect_color',
                                                '/zedA/zed_node_A/right/image_rect_color',
                                                '/zedB/zed_node_B/rgb/image_rect_color',
                                                '/zedB/zed_node_B/right/image_rect_color'],
    '2023-03-02_09-40-54_Reihe 2 schnell.bag': ['/zedA/zed_node_A/rgb/image_rect_color',
                                                '/zedA/zed_node_A/right/image_rect_color',
                                                '/zedB/zed_node_B/rgb/image_rect_color',
                                                '/zedB/zed_node_B/right/image_rect_color'],
    '2023-03-08_14-31-31_Red Prince normal.bag.active': ['/zedA/zed_node_A/rgb/image_rect_color',
                                                         '/zedA/zed_node_A/right/image_rect_color',
                                                         '/zedB/zed_node_B/rgb/image_rect_color',
                                                         '/zedB/zed_node_B/right/image_rect_color'],
    '2023-03-09_09-50-29_Reihe 1 32 Horizontal.bag': ['/zedA/zed_node_A/rgb/image_rect_color',
                                                      '/zedA/zed_node_A/right/image_rect_color',
                                                      '/zedB/zed_node_B/rgb/image_rect_color',
                                                      '/zedB/zed_node_B/right/image_rect_color'],
    '2023-03-09_09-58-53_red Prince H.bag.active': ['/zedA/zed_node_A/rgb/image_rect_color',
                                                    '/zedA/zed_node_A/right/image_rect_color',
                                                    '/zedB/zed_node_B/rgb/image_rect_color',
                                                    '/zedB/zed_node_B/right/image_rect_color'],
    '2023-05-09_15-01-20_D1_komplett.bag': ['/zed_multi/zed2i_left/zed_nodelet_left/left/image_rect_color/compressed',
                                            '/zed_multi/zed2i_left/zed_nodelet_left/right/image_rect_color/compressed',
                                            '/zed_multi/zed2i_right/zed_nodelet_right/left/image_rect_color/compressed',
                                            '/zed_multi/zed2i_right/zed_nodelet_right/right/image_rect_color/compressed'],
    '2023-05-09_15-42-17_D1_komplett.bag': ['/zed_multi/zed2i_left/zed_nodelet_left/left/image_rect_color/compressed',
                                            '/zed_multi/zed2i_left/zed_nodelet_left/right/image_rect_color/compressed',
                                            '/zed_multi/zed2i_right/zed_nodelet_right/left/image_rect_color/compressed',
                                            '/zed_multi/zed2i_right/zed_nodelet_right/right/image_rect_color/compressed'],
    '2023-05-10_09-17-45_Elstar_1_Bluehte.bag': [
        '/zed_multi/zed2i_left/zed_nodelet_left/left/image_rect_color/compressed',
        '/zed_multi/zed2i_left/zed_nodelet_left/right/image_rect_color/compressed',
        '/zed_multi/zed2i_right/zed_nodelet_right/left/image_rect_color/compressed',
        '/zed_multi/zed2i_right/zed_nodelet_right/right/image_rect_color/compressed'],
    '2023-05-10_09-25-16_Sweetango_1_Bluehte.bag': [
        '/zed_multi/zed2i_left/zed_nodelet_left/left/image_rect_color/compressed',
        '/zed_multi/zed2i_left/zed_nodelet_left/right/image_rect_color/compressed',
        '/zed_multi/zed2i_right/zed_nodelet_right/left/image_rect_color/compressed',
        '/zed_multi/zed2i_right/zed_nodelet_right/right/image_rect_color/compressed'],
    '2023-07-03_12-15-42_Fläche d.bag': ['/zed_left/zed_left/left/image_rect_color/compressed',
                                         '/zed_left/zed_left/rgb/image_rect_color/compressed',
                                         '/zed_right/zed_right/left/image_rect_color/compressed',
                                         '/zed_right/zed_right/rgb/image_rect_color/compressed'],
    '2023-07-03_13-05-45_Versuch trocken.bag': ['/zed_left/zed_left/left/image_rect_color/compressed',
                                                '/zed_left/zed_left/rgb/image_rect_color/compressed',
                                                '/zed_right/zed_right/left/image_rect_color/compressed',
                                                '/zed_right/zed_right/rgb/image_rect_color/compressed']
}

matrix = {
    "Left": [1063.1497802734375, 0.0, 990.462646484375, 0.0, 1063.1497802734375, 572.4839477539062, 0.0, 0.0, 1.0],

    "Right": [1079.5146484375, 0.0, 959.8328857421875, 0.0, 1079.5146484375, 552.9612426757812, 0.0, 0.0,
              1.0]}


def print_usage():
    print("Usage: python3 tracking.py Rosbag SVO/Rosbag Left/Right start_frame end_frame True/False row_name")
    print("Please provide arguments separated by spaces.")
    print("Rosbag: Rosbag path for gps data")
    print("SVO/Rosbag: SVO/Rosbag path for images")
    print("Left/Right: Left or Right Camera")
    print("start_frame: first frame number for the apple tree row")
    print("end_frame: last frame number for the apple tree row")
    print("True/False: use gps to reduce false detection")
    print("row_name: name for the apple tree row")


def get_gps(rosbag_file):
    bag = rosbag.Bag(rosbag_file)
    gps = []
    for topic, msg, t in bag.read_messages("/ublox/fix"):
        gps.append([msg.header.stamp.to_nsec(), [msg.latitude, msg.longitude]])
    return gps


def get_frames(svo_rosbag, camera):
    frames = []
    type = svo_rosbag.split(".")[-1]
    if type == "svo":
        svo = svo_rosbag
        init_parameters = sl.InitParameters()
        init_parameters.set_from_svo_file(svo)

        # Open the ZED
        zed = sl.Camera()
        img_mat = sl.Mat()
        depth_mat = sl.Mat()
        err = zed.open(init_parameters)

        while True:
            if zed.grab() == sl.ERROR_CODE.SUCCESS:
                frame_count = zed.get_svo_position()
                timestamp = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).data_ns
                frames.append([timestamp, frame_count])

            elif zed.grab() == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                print("SVO end has been reached. Looping back to first frame")
                break

    else:
        rosbag_name = svo_rosbag.split("/")[-1]
        if camera == "Left":
            topic = camera_topics[rosbag_name][0]
        else:
            topic = camera_topics[rosbag_name][2]
        bag = rosbag.Bag(svo_rosbag)
        for topic, msg, t in bag.read_messages(topic):
            frames.append([msg.header.stamp.to_nsec(), msg])
    return frames


def start_track(synced_list, start_frame, end_frame, tree_tracker, left_right_camera, file_typ, folder_name,row_name,row_line):
    if file_typ == "svo":
        svo = svo_rosbag
        init_parameters = sl.InitParameters()
        init_parameters.set_from_svo_file(svo)
        # Open the ZED
        zed = sl.Camera()
        img_mat = sl.Mat()
        depth_mat = sl.Mat()
        err = zed.open(init_parameters)
    # Create a VideoWriter object
    tree_tracker.set_row(row_name)
    tree_tracker.set_row_line(row_line)
    for frame_and_gps in synced_list[start_frame:end_frame]:
        if file_typ == "svo":
            zed.set_svo_position(frame_and_gps[1])
            zed.grab()
            zed.retrieve_image(img_mat, sl.VIEW.SIDE_BY_SIDE)
            image = img_mat.get_data()
            image = rotate_image(image, left_right_camera)
            image_up, image = np.split(image, 2, axis=0)


        else:
            topic = camera_topics[folder_name][0]
            if "compressed" in topic:
                image = bridge.compressed_imgmsg_to_cv2(frame_and_gps[1])
            else:
                image = bridge.imgmsg_to_cv2(frame_and_gps[1])

            image = rotate_image(image, left_right_camera)
        tree_tracker.update_gps(frame_and_gps[2])
        tree_tracker.update_image(image)
    tree_tracker.finish()


def rotate_image(image, left_right_camera):
    if left_right_camera == "Left":
        rotate = cv2.ROTATE_90_CLOCKWISE
    else:
        rotate = cv2.ROTATE_90_COUNTERCLOCKWISE
    return cv2.rotate(image[..., :3], rotate)


def get_row_line(synced_list, start_frame):
    geod = Geodesic.WGS84
    gps = synced_list[start_frame][2]
    gps2 = synced_list[start_frame + 1][2]
    result = geod.Inverse(gps[0], gps[1], gps2[0], gps2[1])
    driving_direction = result['azi1']
    rows = []
    for filename in os.listdir("geojson"):
        with open(f"geojson/{filename}", 'r') as file:
            data = geojson.load(file)
            if data['type'] == 'FeatureCollection':
                line_strings = [feature for feature in data['features'] if feature['geometry']['type'] == 'LineString']
                for line_string in line_strings:
                    rows.append(line_string["geometry"])
    distance = 99999999
    line = None
    for row in rows:
        line_string = LineString(row["coordinates"])
        point = line_string.coords[0]

        result = geod.Inverse(point[1], point[0], gps[0], gps[1])
        direction = result['azi1']
        if (direction - driving_direction + 360) % 360 < 180:
            current_point = Point(gps[1], gps[0])
            distance_to_line = current_point.distance(line_string)
            if distance_to_line < distance:
                distance = distance_to_line
                line = line_string
    if distance <= 2.5:
        return line
    return None


if len(sys.argv) == 8:
    model = YOLO("YOLOV8WinterSummer.pt")
    bridge = CvBridge()
    rosbag_file = sys.argv[1]
    svo_rosbag = sys.argv[2]
    left_right_camera = sys.argv[3]
    start_frame = int(sys.argv[4])
    end_frame = int(sys.argv[5])
    gps_check = sys.argv[6]
    row_name = sys.argv[7]
    if gps_check.lower() == 'true':
        gps_check = True
    elif gps_check.lower() == 'false':
        gps_check = False

    gps = get_gps(rosbag_file)
    frames = get_frames(svo_rosbag, left_right_camera)
    synced_list = sync.gps_with_image(gps, frames)
    start_index, end_index = sync.clean_indices(synced_list)
    folder_name = svo_rosbag.split("/")[-1]
    file_name = svo_rosbag.split("/")[-1][:19]
    matrix = matrix[left_right_camera]
    matrix[0], matrix[4] = matrix[4], matrix[0]
    matrix[2], matrix[5] = matrix[5], matrix[2]

    intrinsic_matrix = np.array(matrix).reshape(3, 3)

    if start_frame < start_index:
        start_frame = start_index
    if end_frame > end_index:
        end_frame = end_index
    if end_frame == -1:
        print("Images and GPS are not synced", file=sys.stderr)
        sys.exit(0)

    row_line = get_row_line(synced_list, start_frame)

    tree_tracker = Tracker(model, folder_name, file_name, intrinsic_matrix, left_right_camera, gps_check)

    file_typ = svo_rosbag.split(".")[-1]
    start_track(synced_list, start_frame, end_frame, tree_tracker, left_right_camera, file_typ, folder_name,row_name,row_line)

else:
    print("No arguments provided.")
    print_usage()
