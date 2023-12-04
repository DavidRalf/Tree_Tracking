import cv2
import numpy as np
import rosbag
from cv_bridge import CvBridge
import pyzed.sl as sl
from matplotlib import pyplot as plt

import utility.synchronize as sync
from tracker import Tracker
from ultralytics import YOLO

model = YOLO("YOLOV8WinterSummer.pt")
camera = "left"
bridge = CvBridge()
bag = rosbag.Bag("/media/david/T7/2023-07-07_11-26-33_Elstar_1_Laenge_1row.bag")
gps = []
for topic, msg, t in bag.read_messages("/ublox/fix"):
    gps.append([msg.header.stamp.to_nsec(), [msg.latitude, msg.longitude]])

# Set SVO path for playback
svo = "/media/david/T7/2023-07-07_11-26-08_elstar_1_laenge_left_1row.svo"
init_parameters = sl.InitParameters()
init_parameters.set_from_svo_file(svo)

# Open the ZED
zed = sl.Camera()
img_mat = sl.Mat()
depth_mat = sl.Mat()
err = zed.open(init_parameters)

# set svo to the correct position
frame_count = zed.get_svo_position()
frames = []
print("get images")
while True:
    if zed.grab() == sl.ERROR_CODE.SUCCESS:
        # Read side by side frames stored in the SVO
        # zed.retrieve_image(img_mat, sl.VIEW.SIDE_BY_SIDE)
        # image = img_mat.get_data()
        # Get frame count
        frame_count = zed.get_svo_position()
        timestamp = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).data_ns
        frames.append([timestamp, frame_count])

    elif zed.grab() == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
        print("SVO end has been reached. Looping back to first frame")
        break


synced_list_uncleaned = sync.gps_with_image(gps, frames)

synced_list_uncleaned2 = sync.gps_with_image2(gps, frames)

start_index2, end_index2 = sync.clean_indices(synced_list_uncleaned2)

start_index, end_index = sync.clean_indices(synced_list_uncleaned)

print(f"start_index {start_index}")
print(f"end_index {end_index}")
print(f"start_index2 {start_index2}")
print(f"end_index2 {end_index2}")

# Extracting latitude and longitude information
latitudes1 = [entry[2][0] for entry in synced_list_uncleaned[start_index:end_index]]
longitudes1 = [entry[2][1] for entry in synced_list_uncleaned[start_index:end_index]]

# Extracting latitude and longitude information
latitudes2 = [entry[2][0] for entry in synced_list_uncleaned2[start_index2:end_index2]]
longitudes2 = [entry[2][1] for entry in synced_list_uncleaned2[start_index2:end_index2]]

# Plotting latitude and longitude
plt.figure(figsize=(8, 6))
plt.scatter(longitudes1, latitudes1, color='blue', s=100, alpha=0.7)
plt.scatter(longitudes2, latitudes2, color='red', s=100, alpha=0.7)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Scatter Plot of Latitude and Longitude')
plt.grid(True)
plt.show()
