import sys

import rosbag
from rospy import Time, Duration

def downsample_bag(input_bag_path, output_bag_path, target_fps=None):
    with rosbag.Bag(input_bag_path, 'r') as input_bag:
        with rosbag.Bag(output_bag_path, 'w') as output_bag:
            # target_duration = Duration(1.0 / target_fps)
            last_time = None

            for i, (topic, msg, t) in enumerate(input_bag.read_messages()):
                # if last_time is None or t - last_time >= target_duration:
                if i % 6 == 0:
                    output_bag.write(topic, msg, t)
                    last_time = t

if __name__ == "__main__":
    if len(sys.argv) == 3:
        input_bag_path = sys.argv[1]  # 원본 rosbag 파일 경로
        output_bag_path = sys.argv[2]  # 저장할 새로운 rosbag 파일 경로
        # target_fps = 4  # 타겟 FPS

        downsample_bag(input_bag_path, output_bag_path)

    else:
        print("Usage: rosbag_downsample input_bag_path output_bag_path")
