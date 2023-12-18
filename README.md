# SAMSON Tree Tracking


## Project Description
This project was developed as part of the Grundprojekt at HAW (University of Applied Sciences). 
The main task involved isolating an image for each individual tree from a consecutive series of images. This required the detection and tracking of trees across the images.
### Extensions:
- **GPS Integration**: Additional functionality was incorporated by integrating GPS data. This was done to minimize potential misidentifications and to calculate the GPS positions of the recognized trees.


## Table of Contents

- [Installation](#Installation)
- [Usage](#Usage)


## Installation


To run this project, ensure you have Python installed as well as ROS. Then, install the necessary dependencies from the `requirements.txt` file. 

**Please note that the `requirements.txt` contains many packages that are not utilized in this project.**

```bash
# Clone the repository
git clone https://github.com/yourusername/project.git
cd project

# Install dependencies
pip install -r requirements.txt
```
## Usage
### tracking.py
The `tracking.py` script is designed to synchronize GPS and image data from ROS and ROS/SVO and perform tracking for a single row of trees.

To use this script, execute it using Python 3 with the following command:

```bash
python3 tracking.py "/media/david/T71/2023-07-07_11-26-33_Elstar_1_Laenge_1row.bag" "/media/david/T71/2023-07-07_11-26-08_elstar_1_laenge_left_1row.svo" Left 2225 15000 True "Reihe1"
```

#### Arguments
The script requires the following arguments, separated by spaces:

- Rosbag: Path to the ROS bag containing GPS data.
- SVO/Rosbag: Path to the SVO/Rosbag containing images.
- Left/Right: Specify the camera (Left or Right).
- start_frame: The starting frame number for the apple tree row.
- end_frame: The ending frame number for the apple tree row.
- True/False: Use GPS to reduce false detections (True/False).
- row_name: Name for the apple tree row.

#### Additional Information
Please make sure to provide all arguments as described. The script performs tracking based on the given parameters for the specified tree row.

### Standalone
The following Python script demonstrates how the developed tracker can be utilized as a standalone application:
```python
rosbag="2023-07-07_11-26-33_Elstar_1_Laenge_1row.bag"
file_name="2023-07-07_11-26-33"
model = YOLO("YOLOV8WinterSummer.pt")
left_or_right_camera=True or False
gps_check= True or False
tree_tracker = Tracker(model, rosbag, file_name, intrinsic_matrix, left_or_right_camera, gps_check)

for apple_tree_row in dataset:
    row_line = LineString(row) or None # Linestring made from GPS points of the tree row
    tree_tracker.set_row(row_name)
    tree_tracker.set_row_line(row_line)
        for image_and_gps in apple_tree_row:
            gps=image_and_gps[1]
            image=image_and_gps[0]
            tree_tracker.update_gps(gps)
            tree_tracker.update_image(image)
    tree_tracker.finish()
```

#### Useful Methods
- **`make_geojson()`** : generates a GeoJSON file containing the detected trees


- **`get_gps()`** : retrieves a dictionary containing each tree's unique ID as a key and its corresponding GPS locations as values. This method aids in accessing the GPS data associated with detected trees.


- **`show_results()`** : returns the current frame with bounding boxes drawn around detected trees. This functionality is helpful for visualizing the tracking results within the image frames.
