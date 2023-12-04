import matplotlib.pyplot as plt
import numpy as np
from pyproj import CRS, Transformer
from shapely.geometry import Point, LineString


def reproject_point(lon, lat):
    in_crs = CRS.from_epsg(4326)
    out_crs = CRS.from_epsg(4326)
    proj = Transformer.from_crs(in_crs, out_crs, always_xy=True)
    x, y = proj.transform(lon, lat)
    return x, y


def generate_random_points_on_line(number, line):
    # Calculate the step size
    step_size = line.length / (number + 1)

    # Generate evenly spaced points along the line
    evenly_spaced_points = [line.interpolate((i + 1) * step_size) for i in range(number)]

    # while len(points) < number:
    #    point = line.interpolate(uniform(0, line.length))
    #    points.append(point)
    return evenly_spaced_points


def clean_indices(gps_with_image):
    # gps_with_image = [[timestamp, image, [latitude,longitude]]]
    #
    # return start,end,[[timestamp, image, [latitude,longitude]]]
    start_index = 0
    start = False
    end_index = -1
    for index, entry in enumerate(gps_with_image):
        if entry is None and start:
            end_index = index - 1
            break
        if entry is None:
            continue
        if not start:
            start = True
            start_index = index
            continue
        end_index = index
    return start_index, end_index


def gps_with_image(gps_points, images):
    # gps_points = [[timestamp,[latitude,longitude]]
    # images = [[timestamp, frame]]
    # return [[timestamp, image, [latitude,longitude]]]
    gps_points = sorted(gps_points, key=lambda x: x[0])
    images = sorted(images, key=lambda x: x[0])
    old_time = None
    difference = []
    for f in images:
        timestamp = f[0]
        if old_time is None:
            old_time = timestamp
            continue
        difference.append(abs(timestamp - old_time))
        old_time = timestamp

    # Extract timestamps from gps and frames
    frame_timestamps = np.array([entry[0] for entry in images])

    difference = np.mean(difference)

    # Initialize an empty array to store the closest frame for each GPS entry
    synced_gps_and_images = [None] * len(images)

    # Loop through each GPS entry and find the closest frame, ensuring each frame is used only once
    used_frame_indices = set()  # Keep track of used frame indices
    lastid = 0
    for i, (gps_timestamp, gps_coords) in enumerate(gps_points):
        # Calculate the absolute differences between GPS timestamp and available frame timestamps
        abs_diff = np.abs(frame_timestamps - gps_timestamp)
        # Find the index of the closest frame that hasn't been used yet
        closest_indices = [index for index in range(len(images)) if
                           index not in used_frame_indices and abs_diff[index] <= difference]

        if closest_indices:
            closest_index = min(closest_indices, key=lambda idx: abs_diff[idx])
            if closest_index >= lastid:
                used_frame_indices.add(closest_index)  # Mark the frame as used
                synced_gps_and_images[closest_index] = [frame_timestamps[closest_index], images[closest_index][1],
                                                        gps_points[i][1]
                                                        ]
                lastid = closest_index

    # Interpolation
    latest_gps = None
    latest_index = None
    interpolate = False
    point_to_calculate = 0
    for index, list in enumerate(synced_gps_and_images):

        if synced_gps_and_images[index] is None and latest_index is not None:
            interpolate = True
            point_to_calculate += 1
            continue

        if synced_gps_and_images[index] is None and latest_index is None:
            continue
        if interpolate:
            start_point = Point(latest_gps[0], latest_gps[1])
            end_point = Point(synced_gps_and_images[index][2][0], synced_gps_and_images[index][2][1])
            straight_line = LineString([start_point, end_point])
            points = generate_random_points_on_line(point_to_calculate, straight_line)
            for point in points:
                latest_index += 1
                synced_gps_and_images[latest_index] = [frame_timestamps[latest_index], images[latest_index][1],
                                                       [point.x, point.y]
                                                       ]
        latest_gps = synced_gps_and_images[index][2]
        latest_index = index
        point_to_calculate = 0
        continue

    return synced_gps_and_images


def gps_with_image2(gps_points, images):
    # gps_points = [[timestamp,[latitude,longitude]]
    # images = [[timestamp, frame]]
    # return [[timestamp, image, [latitude,longitude]]]
    gps_points = sorted(gps_points, key=lambda x: x[0])
    images = sorted(images, key=lambda x: x[0])
    old_time = None
    difference = []
    for f in images:
        timestamp = f[0]
        if old_time is None:
            old_time = timestamp
            continue
        difference.append(abs(timestamp - old_time))
        old_time = timestamp

    # Extract timestamps from gps and frames
    frame_timestamps = np.array([entry[0] for entry in images])

    difference = np.mean(difference)

    # Initialize an empty array to store the closest frame for each GPS entry
    synced_gps_and_images = [None] * len(images)

    for i, (timestamp, frame) in enumerate(images):
        gps_timestamps = np.array([entry[0] for entry in gps_points])
        abs_diff = gps_timestamps - timestamp
        # Find the indices of the smallest negative and smallest positive numbers
        smallest_negative_index = np.where(abs_diff < 0)[0][np.argmax(abs_diff[abs_diff < 0])]
        smallest_positive_index = np.where(abs_diff > 0)[0][np.argmin(abs_diff[abs_diff > 0])]

        # Get the smallest negative and smallest positive numbers
        smallest_negative = abs_diff[smallest_negative_index]
        smallest_positive = abs_diff[smallest_positive_index]
        print(f"smallest_negative {smallest_negative}")
        print(f"smallest_positive {smallest_positive}")

        # Find the smallest negative number
        #smallest_negative = max(num for num in abs_diff if num < 0)
        # Get the index of value 11
        #smallest_negative_index = abs_diff.index(smallest_negative)
        # Find the smallest positive number
        #smallest_positive = min(num for num in abs_diff if num > 0)
        #smallest_positive_index = abs_diff.index(smallest_positive)
        print(f"smallest_negative {smallest_negative}")
        print(f"smallest_positive {smallest_positive}")

        # Given data
        timestamp_1 = gps_points[smallest_negative_index][0]
        gps_1 = gps_points[smallest_negative_index][1]  # Latitude and Longitude for the first GPS point

        timestamp_2 = gps_points[smallest_positive_index][0]
        gps_2 = gps_points[smallest_positive_index][1]  # Latitude and Longitude for the second GPS point

        image_timestamp = timestamp  # Timestamp of the image

        # Calculate time differences
        time_diff_known = (timestamp_2 - timestamp_1)
        time_diff_image = (image_timestamp - timestamp_1)

        # Calculate proportion of time elapsed
        proportion = time_diff_image / time_diff_known

        # Interpolate latitude and longitude
        interpolated_latitude = gps_1[0] + (gps_2[0] - gps_1[0]) * proportion
        interpolated_longitude = gps_1[1] + (gps_2[1] - gps_1[1]) * proportion
        synced_gps_and_images[i] = [timestamp, frame,
                                    [interpolated_latitude, interpolated_longitude]
                                    ]
        #gps_points.append([timestamp,[interpolated_latitude, interpolated_longitude]])
    return synced_gps_and_images
