#!/usr/bin/env python3

import stretch_funmap.max_height_image as mh
import numpy as np
import scipy.ndimage as nd
import scipy.signal as si
import cv2
import skimage as sk
from skimage.morphology import convex_hull_image
import math
import hello_helpers.hello_misc as hm
import stretch_funmap.navigation_planning as na

from stretch_funmap.numba_height_image import numba_create_segment_image_uint8
import hello_helpers.fit_plane as fp

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Header, UInt8MultiArray
from cv_bridge import CvBridge

callback_received = False
sam_clip_data = []
image_pub = None
bridge = CvBridge()
label_image = []


def detected_coordinates_callback(sam_clip_opt):
    global callback_received, sam_clip_data

    print(type(sam_clip_opt.data))
    callback_received = True
    sam_clip_raw = sam_clip_opt.data

    # sam_clip_data = list(np.array(sam_clip_raw, dtype=np.uint8))
    sam_clip_data = list(np.frombuffer(sam_clip_raw, dtype=np.uint8))

    return sam_clip_data


def init_publish_image():
    # Create a publisher object
    global image_pub

    image_pub = rospy.Publisher('/image_sam', Image, queue_size=10)
    print("Wating fpr 15 sec")
    rospy.sleep(1)  # Sleep for a second to allow for message delivery
    print("Waiting done")


def publish_image(image):
    # Initialize the ROS node
    # rospy.init_node('image_publisher_sam', anonymous=True)
    global image_pub, bridge

    if image_pub == None:
        init_publish_image()

    # Convert the OpenCV image to a ROS Image message
    ros_image = bridge.cv2_to_imgmsg(image, encoding="8UC3")

    # Set a header for the image (timestamp is automatically set)
    ros_image.header = Header()
    ros_image.header.stamp = rospy.Time.now()
    ros_image.header.frame_id = "camera_frame"  # Set the frame ID if required

    # Publish the image just once
    rospy.loginfo("Publishing image...")
    image_pub.publish(ros_image)

    # Sleep briefly to ensure the message is sent before the node shuts down
    print("Image sent!\n\n\n")
    rospy.sleep(0.1)


def get_mask_SAM(height_image):
    h_image = height_image.rgb_image
    # cv2.imwrite("/home/hello-robot/Documents/lmnavgrasp/h_image.jpg", h_image)
    # cv2.imwrite("/home/hello-robot/Documents/lmnavgrasp/rgb_image.jpg", height_image.rgb_image)

    #! Sending an image
    for i in range(2):
        # height_image.rgb_image
        publish_image(h_image)
        rospy.sleep(1)

    rospy.Subscriber('/detected_coordinates', UInt8MultiArray,
                     detected_coordinates_callback)
    rospy.sleep(2)  # Sleep for a second to allow for message delivery

    rate = rospy.Rate(10)  # 10 Hz
    cnt = 0
    while not callback_received and not rospy.is_shutdown() and cnt < 10:
        print("waiting")
        cnt += 1
        rate.sleep()

        key = cv2.waitKey(1)  # 1 millisecond wait
        if key == 27:  # 27 is the ASCII value for the 'Esc' key
            print("Esc key pressed, exiting loop")
            break

    sample_coordinates = [
        (217, 34), (218, 34), (219, 34), (220, 34), (221, 34), (222, 34), (225, 34),
        (217, 35), (218, 35), (219, 35), (220, 35), (221, 35), (222, 35), (223, 35),
        (224, 35), (225, 35), (226, 35), (216, 36), (217, 36), (218, 36), (219, 36),
        (220, 36), (221, 36), (222, 36), (223, 36), (224, 36), (225, 36), (226, 36),
        (227, 36), (217, 37), (218, 37), (219, 37), (220, 37), (221, 37), (222, 37),
        (223, 37), (224, 37), (225, 37), (226, 37), (227, 37), (218, 38), (219, 38),
        (220, 38), (221, 38), (222, 38), (223, 38), (224, 38), (225, 38), (226, 38),
        (218, 39), (219, 39), (220, 39), (221, 39), (222, 39), (223, 39), (224, 39),
        (225, 39), (226, 39), (218, 40), (219, 40), (220, 40), (221, 40), (222, 40),
        (223, 40), (224, 40), (225, 40), (226, 40), (218, 41), (219, 41), (220, 41),
        (221, 41), (222, 41), (223, 41), (224, 41), (225, 41), (226, 41), (218, 42),
        (219, 42), (220, 42), (221, 42), (222, 42), (223, 42), (224, 42), (225, 42),
        (226, 42), (218, 43), (222, 43), (223, 43), (224, 43), (225, 43), (226, 43),
        (223, 44), (224, 44), (225, 44), (226, 44), (224, 45), (225, 45), (226, 45)
    ]
    sample_2 = [
        [207, 71],
        [208, 71],
        [208, 70],
        [211, 70],
        [209, 70],
        [210, 70],
        [210, 70],
        [210, 70],
        [211, 70],
        [212, 70],
        [213, 70],
        [214, 70],
        [215, 70],
        [214, 71],
        [214, 72],
        [216, 72],
        [217, 73],
        [216, 74],
        [216, 76],
        [215, 77],
        [216, 77],
        [216, 77],
        [216, 75],
        [215, 74],
        [215, 74],
        [215, 74],
        [215, 74],
        [215, 73],
        [214, 73],
        [213, 73],
        [212, 75],
        [212, 75],
        [211, 74],
        [211, 73],
        [211, 72],
        [211, 72],
        [210, 72],
        [209, 72],
        [208, 74],
        [208, 74],
        [208, 75],
        [208, 77],
        [208, 77],
        [212, 78],
        [212, 77],
        [211, 77],
        [211, 77],
        [208, 79],
        [208, 79],
        [213, 79],
        [213, 79],
        [215, 79],
        [215, 79],
        [216, 79],
        [217, 78],
        [217, 78],
        [217, 78],
        [217, 76],
        [217, 76],
        [217, 76],
        [217, 76],
        [217, 75],
        [217, 73],
        [215, 72],
        [215, 71],
        [214, 70],
        [214, 70],
        [213, 71],
        [211, 72],
        [210, 73],
        [209, 74],
        [209, 74],
        [204, 78],
        [204, 79],
        [201, 79],
        [208, 74],
        [208, 73],
        [211, 70],
        [213, 71],
        [213, 71],
        [216, 75],
        [215, 78],
        [214, 76],
        [215, 73],
        [215, 72],
        [213, 73],
        [214, 77]
    ]

    if callback_received:
        print('\n\n\n\n\n')

        print(30*'*')
        print("Published image and received the data")
        print(sam_clip_data)
        print(30*'*')

        reshape_sam_clip = np.reshape(sam_clip_data, h_image.shape[:2])

        label_image = np.asarray(reshape_sam_clip, dtype=int)
        print("len of labels", np.unique(label_image))

    else:
        # use the current image
        print("No image received so using the current live image")

        # Create the mask
        mask = np.zeros((334, 434), dtype=np.uint8)
        for x, y in sample_2:
            if 0 <= y < 334 and 0 <= x < 434:
                mask[y, x] = 255

        # Convert mask to 3 channels
        binary_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # Apply the mask
        label_image = cv2.bitwise_and(h_image, binary_mask)

    cv2.imwrite(
        "/home/hello-robot/Documents/lmnavgrasp/label_image.jpg", label_image)


# ! To change
def find_object_to_grasp(height_image, display_on=False, object_bbox=None):

    # RUN SAM
    get_mask_SAM(height_image)

    h_image = height_image.image
    m_per_unit = height_image.m_per_height_unit
    m_per_pix = height_image.m_per_pix
    height, width = height_image.image.shape
    # print("dimensions", height, width) ->  334 434
    robot_xy_pix = [width/2, 0]
    surface_mask, plane_parameters = find_closest_flat_surface(
        height_image, robot_xy_pix, display_on=False)

    if surface_mask is None:
        if display_on:
            print('No elevated surface found.')
        print("No surface mask")
        return None

    surface_height_pix = np.max(h_image[surface_mask > 0])
    surface_height_m = m_per_unit * surface_height_pix
    height_image.apply_planar_correction(plane_parameters, surface_height_pix)
    h_image = height_image.image
    if display_on:
        cv2.imshow('corrected height image', h_image)
        cv2.imshow('rgb image', height_image.rgb_image)

    if display_on:
        rgb_image = height_image.rgb_image.copy()
        rgb_image[surface_mask > 0] = (
            rgb_image[surface_mask > 0]/2) + [0, 127, 0]

    #####################################
    # Select candidate object points

    # Define the minimum height for a candidate object point
    min_object_height_m = 0.01
    min_obstacle_height_m = surface_height_m + min_object_height_m
    min_obstacle_height_pix = min_obstacle_height_m / m_per_unit

    # Define the maximum height for a candidate object point
    # from HeadScan in mapping.py and ManipulationView in manipulation_planning.py)
    robot_camera_height_m = 1.13
    voi_safety_margin_m = 0.02
    max_object_height_m = 0.4
    max_obstacle_height_m = min(robot_camera_height_m - voi_safety_margin_m,
                                surface_height_m + max_object_height_m)
    max_obstacle_height_pix = max_obstacle_height_m / m_per_unit

    # Select candidate object points that are within the valid height range
    obstacle_selector = (h_image > min_obstacle_height_pix) & (
        h_image < max_obstacle_height_pix)
    # print("obstacle_selector", np.argwhere(obstacle_selector))

    obstacle_mask = np.uint8(obstacle_selector)

    # Find the convex hull of the surface points to represent the full
    # surface, overcoming occlusion holes, noise, and other phenomena.
    surface_convex_hull_mask = convex_hull_image(surface_mask)

    # Select candidate object points that are both within the valid
    # height range and on the surface
    # ? This just choses what part of the image to see
    # print("surface_convex_hull_mask", np.argwhere(surface_convex_hull_mask))
    obstacles_on_surface_selector = (
        obstacle_selector & surface_convex_hull_mask)
    # print("obstacles_on_surface_selector", np.argwhere(obstacles_on_surface_selector))
    obstacles_on_surface = np.uint8(255.0 * obstacles_on_surface_selector)

    # Dilate and erode the candidate object points to agglomerate
    # object parts that might be separated due to occlusion, noise,
    # and other phenomena.
    kernel_width_pix = 5  # 3
    iterations = 3  # 5
    kernel_radius_pix = (kernel_width_pix - 1) // 2
    kernel = np.zeros((kernel_width_pix, kernel_width_pix), np.uint8)
    cv2.circle(kernel, (kernel_radius_pix, kernel_radius_pix),
               kernel_radius_pix, 255, -1)
    use_dilation = True
    if use_dilation:
        obstacles_on_surface = cv2.dilate(
            obstacles_on_surface, kernel, iterations=iterations)
    use_erosion = True
    if use_erosion:
        obstacles_on_surface = cv2.erode(
            obstacles_on_surface, kernel, iterations=iterations)

    #####################################
    # Process the candidate object points

    # Treat connected components of candidate object points as objects. Fit ellipses to these segmented objects.
    if label_image == []:
        label_image, max_label_index = sk.measure.label(
            obstacles_on_surface, background=0, return_num=True, connectivity=2)

    # inverted
    color_map = {
        0: [0, 0, 0],          # Background color (black)
        1: [255, 0, 0],        # Object 1 color (red)
        2: [0, 255, 0],        # Object 2 color (green)
        3: [0, 0, 255],        # Object 3 color (blue)
        4: [100, 100, 100]
    }

    if callback_received:
        reshape_sam_clip = np.reshape(sam_clip_data, h_image.shape)

        # !Setting the label_image
        label_image = np.asarray(reshape_sam_clip, dtype=int)
        print("len of labels", np.unique(label_image))
    else:
        print('\n\n\n\n no image')
        label_image = label_image

    # Create a new colored image
    colored_image = np.zeros(
        (label_image.shape[0], label_image.shape[1], 3), dtype=int)

    # Apply the predefined color map
    for label_idx in range(len(color_map)):
        for x in range(label_image.shape[0]):
            for y in range(label_image.shape[1]):
                if label_idx == label_image[x, y]:
                    colored_image[x, y] = color_map[label_idx]

        if label_idx in color_map:
            colored_image[label_image == label_idx] = color_map[label_idx]

    region_properties = sk.measure.regionprops(
        label_image, intensity_image=None, cache=True)
    print("Done region prop")
    if display_on:
        rgb_image = height_image.rgb_image.copy()
        color_label_image = sk.color.label2rgb(
            label_image, image=rgb_image, colors=None, alpha=0.3, bg_label=0, bg_color=(0, 0, 0), image_alpha=1, kind='overlay')
        cv2.imshow('color_label_image', color_label_image)

    # Proceed if an object was found.
    if len(region_properties) > 0:

        # Select the object with the largest area.
        largest_region = None
        largest_area = 0.0
        for region in region_properties:
            if region.area > largest_area:
                largest_region = region
                largest_area = region.area

        # Make the object with the largest area the grasp target. In
        # the future, other criteria could be used, such as the
        # likelihood that the gripper can actually grasp the
        # object. For example, the target object might be too large.
        object_region = largest_region

        # Collect and compute various features for the target object.
        object_ellipse = get_ellipse(object_region)
        object_area_m_sqr = object_region.area * pow(m_per_pix, 2)
        min_row, min_col, max_row, max_col = object_region.bbox
        object_bounding_box = {
            'min_row': min_row, 'min_col': min_col, 'max_row': max_row, 'max_col': max_col}

        # Only compute height statistics using the original,
        # high-confidence heights above the surface that are a part of
        # the final object region.
        # print("label_image", np.argwhere(label_image))
        object_selector = (label_image == object_region.label)
        # print("object selector", np.argwhere(object_selector))
        # print("obstacles_on_surface_selector", np.argwhere(obstacles_on_surface_selector))

        # obstacles_on_surface_selector &  #! Should set the & when i get the actual coordinates with SAM
        object_height_selector = object_selector
        # object_height_selector = object_selector & obstacles_on_surface_selector

        right = h_image[object_height_selector]
        # np.savetxt('/home/hello-robot/Downloads/test_lmgraspnav/' + 'h_image.csv', h_image, delimiter=',', fmt='%.6f')
        # np.savetxt('/home/hello-robot/Downloads/test_lmgraspnav/'+'object_height_selector.csv', object_height_selector, delimiter=',', fmt='%.6f')

        # print("h_image", h_image.shape, len(h_image == True))
        # print("right", right)
        # print("m_per_unit", m_per_unit)
        # print("surface_height_m", surface_height_m)
        object_heights_m = (m_per_unit * right) - surface_height_m
        # print("object_heights_m", object_heights_m)

        object_mean_height_m = np.mean(object_heights_m)
        # print("object_mean_height_m", object_mean_height_m)

        object_max_height_m = np.max(object_heights_m)
        object_min_height_m = np.min(object_heights_m)
        # print("object selector", object_mean_height_m, object_selector)

        if display_on:
            print('object max height = {0} cm'.format(
                object_max_height_m * 100.0))
            print('object mean height = {0} cm'.format(
                object_mean_height_m * 100.0))
            rgb_image = height_image.rgb_image.copy()
            # rgb_image[surface_convex_hull_mask > 0] = (rgb_image[surface_convex_hull_mask > 0]/2) + [0, 127, 0]
            rgb_image[surface_convex_hull_mask > 0] = (
                rgb_image[surface_convex_hull_mask > 0]//2) + [0, 127, 0]
            rgb_image[label_image == object_region.label] = [0, 0, 255]
            draw_ellipse_axes_from_region(
                rgb_image, largest_region, color=[255, 255, 255])
            cv2.imshow('object to grasp', rgb_image)

        # ellipse = {'centroid': centroid,
        #            'minor': {'axis': minor_axis, 'length': r.minor_axis_length},
        #            'major': {'axis': major_axis, 'length': r.major_axis_length, 'ang_rad': major_ang_rad}}

        # Prepare grasp target information.
        grasp_location_xy_pix = object_ellipse['centroid']
        major_length_pix = object_ellipse['major']['length']
        major_length_m = m_per_pix * major_length_pix
        minor_length_pix = object_ellipse['minor']['length']
        diff_m = m_per_pix * (major_length_pix - minor_length_pix)

        print("grasp loc of ellipse:",
              grasp_location_xy_pix, diff_m, object_ellipse)

        if display_on:
            print('object_ellipse =', object_ellipse)
        max_gripper_aperture_m = 0.08
        if (diff_m > 0.02) or (major_length_m > max_gripper_aperture_m):
            grasp_elongated = True
            grasp_width_pix = minor_length_pix
            grasp_aperture_axis_pix = object_ellipse['minor']['axis']
            grasp_long_axis_pix = object_ellipse['major']['axis']
            print("True")
        else:
            grasp_elongated = False
            grasp_width_pix = major_length_pix
            grasp_aperture_axis_pix = None
            grasp_long_axis_pix = None
            print("False")

        grasp_width_m = m_per_pix * grasp_width_pix

        fingertip_diameter_m = 0.03
        grasp_location_above_surface_m = max(
            0.0, object_mean_height_m - (fingertip_diameter_m/2.0))
        grasp_location_z_pix = surface_height_pix + \
            (grasp_location_above_surface_m / m_per_unit)

        # print("grasp loc z_pix:", grasp_location_z_pix)
        max_object_height_above_surface_m = object_max_height_m

        grasp_target = {'location_xy_pix': grasp_location_xy_pix,
                        'elongated': grasp_elongated,
                        'width_pix': grasp_width_pix,
                        'width_m': grasp_width_m,
                        'aperture_axis_pix': grasp_aperture_axis_pix,
                        'long_axis_pix': grasp_long_axis_pix,
                        'location_above_surface_m': grasp_location_above_surface_m,
                        'location_z_pix': grasp_location_z_pix,
                        'object_max_height_above_surface_m': object_max_height_m,
                        'surface_convex_hull_mask': surface_convex_hull_mask,
                        'object_selector': object_selector,
                        'object_ellipse': object_ellipse}
        print(20*'^')
        print("grasp loc:", grasp_target, "\n\n\n")

        if display_on:
            print('_________________________________')
            print('grasp_target =')
            print(grasp_target)
            print('_________________________________')

        return grasp_target


def find_closest_flat_surface(height_image, robot_xy_pix, display_on=False):
    # height_image is the actual image
    h = height_image

    height, width = h.image.shape
    robot_xy_pix = [width/2, 0]

    best_surface = None
    a = None

    if display_on:
        color_im = np.zeros((height, width, 3), np.uint8)
        color_im[:, :, 0] = h.image
        color_im[:, :, 1] = h.image
        color_im[:, :, 2] = h.image

    distance_map = False
    traversable_mask = False

    # segment the max height image
    image = h.image
    m_per_unit = h.m_per_height_unit
    m_per_pix = h.m_per_pix

    # This assumes that the max_height_image is parallel to the
    # floor and that the z component of the volume of interest's
    # origin is defined with respect to a frame_id (f) for which
    # z_f = 0.0 corresponds with the modeled ground plane.
    zero_height = -h.voi.origin[2]
    image_rgb = h.rgb_image

    # ~0.1 or ~0.2 for tabletop
    # ~0.3 for room
    segmentation_scale = 0.1

    # !Change Here!!!!!
    # segments_image, segment_info, height_to_segment_id = segment(image, m_per_unit, zero_height, segmentation_scale, verbose=False)
    # if segment_info is None:
    #     return None, None

    # floor_id, floor_mask = find_floor(segment_info, segments_image, verbose=False)
    # if floor_mask is None:
    #     return None, None

    # remove_floor = True
    # if remove_floor:
    #     segments_image[floor_mask > 0] = 0

    # label_image, max_label_index = sk.measure.label(segments_image, background=0, return_num=True, connectivity=2)
    region_properties = sk.measure.regionprops(
        label_image, intensity_image=image, cache=True)
    if display_on:
        color_label_image = sk.color.label2rgb(
            label_image, image=image_rgb, colors=None, alpha=0.3, bg_label=0, bg_color=(0, 0, 0), image_alpha=1, kind='overlay')

    # manipulation surface area category
    # large
    min_area_m_sqr = 4.0 * (0.1 * 0.1)  # 4 x (10 cm squared regions)
    # reachable height
    min_height_m = 0.0
    max_height_m = 0.92

    surfaces = []

    for region in region_properties:
        # area : "Number of pixels of the region."
        label = region.label
        area_m_sqr = region.area * pow(m_per_pix, 2)
        mean_height_m = region.mean_intensity * m_per_unit
        max_height_m = region.max_intensity * m_per_unit
        min_height_m = region.min_intensity * m_per_unit
        yc, xc = region.centroid
        yc = int(round(yc))
        xc = int(round(xc))
        min_row, min_col, max_row, max_col = region.bbox

        if (area_m_sqr >= min_area_m_sqr) and (mean_height_m > min_height_m) and (mean_height_m < max_height_m):
            # Likely a reachable surface region
            surfaces.append({'label': label,
                             'area_m_sqr': area_m_sqr,
                             'mean_height_m': mean_height_m,
                             'max_height_m': max_height_m,
                             'min_height_m': min_height_m,
                             'centroid_x': xc,
                             'centroid_y': yc,
                             'bbox': {'min_row': min_row, 'min_col': min_col, 'max_row': max_row, 'max_col': max_col}})

    surface_label_image = np.zeros_like(label_image, np.uint8)
    for s in surfaces:
        i = s['label']
        surface_label_image[label_image == i] = i
    if display_on:
        surface_label_image_rgb = sk.color.label2rgb(
            surface_label_image, image=None, colors=None, alpha=0.3, bg_label=0, bg_color=(0, 0, 0), image_alpha=1, kind='overlay')

        best_surface_image_rgb = np.zeros(
            surface_label_image_rgb.shape, np.uint8)

    if len(surfaces) > 0:
        # Find the surface that is closest to the robot, but
        # is not the robot's own body.

        height, width = surface_label_image.shape
        robot_xy_pix = [width/2, 0]
        robot_loc = np.array(robot_xy_pix)
        nearest_x, nearest_y, nearest_surface_label = hm.find_nearest_nonzero(
            surface_label_image, robot_loc)

        best_surface = np.uint8(surface_label_image == nearest_surface_label)

        #####################

        original_best_surface = best_surface.copy()

        a, X, z = fp.fit_plane_to_height_image(image, best_surface)
        fit_error, z_fit = fp.fit_plane_to_height_image_error(a, X, z)
        fit_error = np.abs(fit_error)
        original_fit_error = fit_error.copy()
        fit_error_m = fit_error * m_per_unit

        min_error_m = np.min(fit_error_m)
        max_error_m = np.max(fit_error_m)
        mean_error_m = np.mean(fit_error_m)
        if display_on:
            print('-- first fit errors --')
            print('min_error_m =', min_error_m)
            print('max_error_m =', max_error_m)
            print('mean_error_m =', mean_error_m)

        surface_error_threshold_m = 0.02
        surface_points = fit_error_m < surface_error_threshold_m
        surface_temp = best_surface[best_surface > 0]
        surface_temp[~surface_points] = 0
        best_surface[best_surface > 0] = surface_temp

        fit_twice = True
        if fit_twice:
            a, X, z = fp.fit_plane_to_height_image(image, best_surface)
            fit_error, z_fit = fp.fit_plane_to_height_image_error(a, X, z)
            fit_error = np.abs(fit_error)
            fit_error_m = fit_error * m_per_unit

            min_error_m = np.min(fit_error_m)
            max_error_m = np.max(fit_error_m)
            mean_error_m = np.mean(fit_error_m)
            if display_on:
                print('-- second fit errors --')
                print('min_error_m =', min_error_m)
                print('max_error_m =', max_error_m)
                print('mean_error_m =', mean_error_m)

            surface_error_threshold_m = 0.01
            surface_points = fit_error_m < surface_error_threshold_m
            surface_temp = best_surface[best_surface > 0]
            surface_temp[~surface_points] = 0
            best_surface[best_surface > 0] = surface_temp

        if display_on:
            original_max_error = np.max(original_fit_error)
            original_min_error = np.min(original_fit_error)
            display_fit_error = np.uint8(
                255.0 * ((original_fit_error - original_min_error) / (original_max_error - original_min_error)))
            colors = np.zeros((display_fit_error.shape[0], 3), np.uint8)
            colors[:, 2] = display_fit_error
            best_surface_image_rgb[surface_label_image ==
                                   nearest_surface_label] = colors
            best_surface_image_rgb[best_surface > 0] = [255, 0, 0]

            radius = 5
            # draw a green circle at the robot's location
            cv2.circle(best_surface_image_rgb,
                       (robot_loc[0], robot_loc[1]), radius, [0, 255, 0], 2)
            # draw a white circle at the point closest to the robot
            cv2.circle(best_surface_image_rgb, (nearest_x,
                       nearest_y), radius, [255, 255, 255], 2)

    plane_mask = best_surface
    plane_parameters = a
    return plane_mask, plane_parameters


def get_ellipse(region_properties):
    # calculate line segments for ellipse axes
    r = region_properties
    centroid_y, centroid_x = r.centroid
    major_ang_rad = r.orientation

    minor_offset_x = (0.5 * math.sin(major_ang_rad) * r.minor_axis_length)
    minor_offset_y = (0.5 * math.cos(major_ang_rad) * r.minor_axis_length)
    minor_1_x = centroid_x - minor_offset_x
    minor_2_x = centroid_x + minor_offset_x
    minor_1_y = centroid_y - minor_offset_y
    minor_2_y = centroid_y + minor_offset_y

    major_offset_x = (0.5 * math.cos(major_ang_rad) * r.major_axis_length)
    major_offset_y = (0.5 * math.sin(major_ang_rad) * r.major_axis_length)
    major_1_x = centroid_x + major_offset_x
    major_2_x = centroid_x - major_offset_x
    major_1_y = centroid_y - major_offset_y
    major_2_y = centroid_y + major_offset_y

    centroid = (centroid_x, centroid_y)
    minor_axis = ((minor_1_x, minor_1_y), (minor_2_x, minor_2_y))
    major_axis = ((major_1_x, major_1_y), (major_2_x, major_2_y))

    ellipse = {'centroid': centroid,
               'minor': {'axis': minor_axis, 'length': r.minor_axis_length},
               'major': {'axis': major_axis, 'length': r.major_axis_length, 'ang_rad': major_ang_rad}}

    return ellipse


def draw_ellipse_axes(image, ellipse, color=[255, 255, 255], draw_line_contrast=True):
    minor_axis = np.int32(np.round(np.array(ellipse['minor']['axis'])))
    major_axis = np.int32(np.round(np.array(ellipse['major']['axis'])))
    if draw_line_contrast:
        cv2.line(image, tuple(minor_axis[0]),
                 tuple(minor_axis[1]), [0, 0, 0], 3)
        cv2.line(image, tuple(major_axis[0]),
                 tuple(major_axis[1]), [0, 0, 0], 3)
    cv2.line(image, tuple(minor_axis[0]), tuple(minor_axis[1]), color, 1)
    cv2.line(image, tuple(major_axis[0]), tuple(major_axis[1]), color, 1)


def draw_ellipse_axes_from_region(image, region_properties, color=[255, 255, 255], draw_line_contrast=True):
    ellipse = get_ellipse(region_properties)
    minor_axis = np.int32(np.round(np.array(ellipse['minor']['axis'])))
    major_axis = np.int32(np.round(np.array(ellipse['major']['axis'])))
    if draw_line_contrast:
        cv2.line(image, tuple(minor_axis[0]),
                 tuple(minor_axis[1]), [0, 0, 0], 3)
        cv2.line(image, tuple(major_axis[0]),
                 tuple(major_axis[1]), [0, 0, 0], 3)
    cv2.line(image, tuple(minor_axis[0]), tuple(minor_axis[1]), color, 1)
    cv2.line(image, tuple(major_axis[0]), tuple(major_axis[1]), color, 1)
