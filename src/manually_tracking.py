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
bag = rosbag.Bag("/media/david/T71/2023-07-07_11-26-33_Elstar_1_Laenge_1row.bag")
gps = []
for topic, msg, t in bag.read_messages("/ublox/fix"):
    gps.append([msg.header.stamp.to_nsec(), [msg.latitude, msg.longitude]])

# Set SVO path for playback
svo = "/media/david/T71/2023-07-07_11-26-08_elstar_1_laenge_left_1row.svo"
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

synced_list_uncleaned = sync.gps_with_image2(gps, frames)
start_index, end_index = sync.clean_indices(synced_list_uncleaned)

if camera == "left":
    rotate = cv2.ROTATE_90_CLOCKWISE
else:
    rotate = cv2.ROTATE_90_COUNTERCLOCKWISE

folder_name = svo.split("/")[-1]
file_name = svo.split("/")[-1][:19]

matrinx = [1063.1497802734375, 0.0, 990.462646484375, 0.0, 1063.1497802734375, 572.4839477539062, 0.0, 0.0, 1.0]
matrinx[0], matrinx[4] = matrinx[4], matrinx[0]
matrinx[2], matrinx[5] = matrinx[5], matrinx[2]
intrinsic_matrix = np.array(matrinx).reshape(3, 3)

tree_tracker = Tracker(model, folder_name, file_name, intrinsic_matrix, "Left")

start_index = 2225
for frame_and_gps in synced_list_uncleaned[start_index:end_index]:
    zed.set_svo_position(frame_and_gps[1])
    zed.grab()
    zed.retrieve_image(img_mat, sl.VIEW.SIDE_BY_SIDE)
    image = img_mat.get_data()
    image = cv2.rotate(image[..., :3], rotate)
    image_up, image_down = np.split(image, 2, axis=0)
    tree_tracker.update_gps(frame_and_gps[2])
    tree_tracker.update_image(image_down)
    frame_new = tree_tracker.show_results()
    cv2.imshow("Frame", frame_new)
    cv2.waitKey(0)

tree_positions = tree_tracker.tree_3d_positions
x = []
y = []
# Extract y-values (tree indices) and x-values from the tree positions
for idx, tree in enumerate(tree_positions):
    y.append(tree_positions[tree][1])  # Adding tree index
    x.append(tree_positions[tree][0])  # x-coordinate of the tree

# Plotting x-coordinates against tree indices
plt.figure(figsize=(8, 6))
plt.plot(x, y, linestyle='-', marker='o', color='green')  # Line plot of x-coordinates vs. tree indices
plt.ylabel('lon-coordinate')
plt.xlabel('lat-coordinate')
plt.title('X-coordinates vs. Tree Indices')
plt.grid(True)
plt.show()

tree_tracker.finish()
tree_tracker.reset()
