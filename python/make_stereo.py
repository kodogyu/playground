import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import numpy as np
import cv2
import matplotlib.pyplot as plt

from copy import deepcopy

# Rosbags
read_bag = rosbag.Bag('../test_rosbags/l515_imu_synchronized.bag')
write_bag = rosbag.Bag('../test_rosbags/l515_imu_synchronized_stereo.bag', 'w')
# Basic information
focal_length = 604  # f_x = 603.838, f_y = 604.019
baseline = 10  # 10mm

got_color_image = False
got_depth_image = False
# Read rosbag
i = 0
for topic, message, t in read_bag.read_messages(topics=['/camera/color/image_raw', '/camera/depth/image_rect_raw', '/camera/imu']):
    print("processing..." + str(i))
    i+=1
    if topic == "/camera/color/image_raw":
        color_image_message = message
        got_color_image = True
    elif topic == "/camera/depth/image_rect_raw":
        depth_image_message = message
        got_depth_image = True
    else:
        write_bag.write(topic, message, t)

    if got_color_image and got_depth_image:
        bridge = CvBridge()
        color_image = bridge.imgmsg_to_cv2(color_image_message)
        depth_image = bridge.imgmsg_to_cv2(depth_image_message)
        right_image = np.zeros_like(color_image)

        # Calculate the right image
        for row in range(480):
            for col in range(640):
                depth = depth_image[row, col]  # mm
                if depth:
                    disparity = focal_length * baseline // depth
                    
                    if (col - disparity > 0):
                        right_image[row, col - disparity] = color_image[row, col]
        # Make ros message
        right_image_message = bridge.cv2_to_imgmsg(right_image, "rgb8")
        right_image_message.header = deepcopy(color_image_message.header)
        right_image_message.header.frame_id = "camera_color_optical_right_frame"

        write_bag.write('/camera/color/image_raw', color_image_message)
        write_bag.write('/camera/color_right/image_raw', right_image_message)
        write_bag.write('/camera/depth/image_rect_raw', depth_image_message)

read_bag.close()
write_bag.close()