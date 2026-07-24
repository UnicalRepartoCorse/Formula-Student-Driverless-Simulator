import cv2
import rosbag2_py
from cv_bridge import CvBridge
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image

bag_path = 'my_recording'          # Path to bag folder
topic_name = '/Stereo/predicted_image'  # Topic name
output_mp4 = 'realtime_output.mp4'

reader = rosbag2_py.SequentialReader()
# Change storage_id to 'mcap' or 'sqlite3' depending on your bag type
storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id='mcap')
converter_options = rosbag2_py.ConverterOptions(
    input_serialization_format='cdr',
    output_serialization_format='cdr'
)
reader.open(storage_options, converter_options)
reader.set_filter(rosbag2_py.StorageFilter(topics=[topic_name]))

bridge = CvBridge()

# Pass 1: Find timestamps to calculate exact playback duration & overall FPS
timestamps = []
print("Scanning bag timestamps...")
while reader.has_next():
    (topic, data, t) = reader.read_next()
    timestamps.append(t)

if not timestamps:
    print("No messages found!")
    exit()

duration_sec = (timestamps[-1] - timestamps[0]) / 1e9
total_frames = len(timestamps)
calculated_fps = total_frames / duration_sec
print(f"Total frames: {total_frames} over {duration_sec:.2f} seconds.")
print(f"Actual average capture FPS: {calculated_fps:.2f}")

# Reset reader for Pass 2: Extract video at calculated FPS
reader = rosbag2_py.SequentialReader()
reader.open(storage_options, converter_options)
reader.set_filter(rosbag2_py.StorageFilter(topics=[topic_name]))

video_writer = None
while reader.has_next():
    (topic, data, t) = reader.read_next()
    msg = deserialize_message(data, Image)
    cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    
    # Pad odd dimensions to even numbers automatically
    h, w, _ = cv_img.shape
    new_h = h if h % 2 == 0 else h + 1
    new_w = w if w % 2 == 0 else w + 1
    
    if (new_h != h) or (new_w != w):
        cv_img = cv2.copyMakeBorder(cv_img, 0, new_h - h, 0, new_w - w, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        
    if video_writer is None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_mp4, fourcc, calculated_fps, (new_w, new_h))
        
    video_writer.write(cv_img)

if video_writer:
    video_writer.release()
    print(f"Saved real-time video to '{output_mp4}'")
