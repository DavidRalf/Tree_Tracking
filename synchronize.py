import numpy as np


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

    synced_gps_and_images = [None] * len(images)

    for i, (timestamp, frame) in enumerate(images):
        gps_timestamps = np.array([entry[0] for entry in gps_points])
        abs_diff = gps_timestamps - timestamp

        negative_indices = np.where(abs_diff < 0)[0]
        positive_indices = np.where(abs_diff > 0)[0]
        if negative_indices.size > 0:
            smallest_negative_index = negative_indices[np.argmax(abs_diff[negative_indices])]
        else:
            continue

        if positive_indices.size > 0:
            smallest_positive_index = positive_indices[np.argmin(abs_diff[positive_indices])]
        else:
            continue

        timestamp_1 = gps_points[smallest_negative_index][0]
        gps_1 = gps_points[smallest_negative_index][1]

        timestamp_2 = gps_points[smallest_positive_index][0]
        gps_2 = gps_points[smallest_positive_index][1]

        image_timestamp = timestamp

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
    return synced_gps_and_images
