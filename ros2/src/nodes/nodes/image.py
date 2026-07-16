import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry
from vision_msgs.msg import Detection2DArray, Detection2D
from cv_bridge import CvBridge
import cv2
import numpy as np
import sys
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import queue
import tf2_ros

venv_path = "/home/lenovo/Formula-Student-Driverless-Simulator/ros_env/lib/python3.12/site-packages"
if venv_path in sys.path:
    sys.path.remove(venv_path)
sys.path.insert(0, venv_path)

#from . import inject
from ultralytics import YOLO
import ultralytics.nn.modules as modules
import ultralytics.nn.tasks as tasks

#modules.DW = inject.DW
#tasks.DW = inject.DW

#modules.CDW = inject.CDW
#tasks.CDW = inject.CDW


def get_camera_values(node):
    node.fx, node.fy = 392.5, 392.5, 392.5, 392.5


class ImageListener(Node):
    def __init__(self, test=True):
        super().__init__('image_listener')

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.test=test

        get_camera_values(self)

        self.queue = queue.Queue(10)
        self.detection_group = MutuallyExclusiveCallbackGroup()
        self.processing_group = MutuallyExclusiveCallbackGroup()

        model_path = "/home/lenovo/Formula-Student-Driverless-Simulator/ros2/src/nodes/nodes/custom_b_openvino_model"
        try:
            self.model = YOLO(model_path)
            #self.model.to("cuda")

            self.get_logger().info(f'Modello caricato correttamente da {model_path}')
        except Exception as e:
            self.get_logger().error(f'Errore caricamento modello: {str(e)}')
            self.destroy_node()
            raise SystemExit("Model failed to load, shutting down node.")
        
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            '/fsds/Stereo/image_color',
            self.listener_callback,
            10,
            callback_group = self.detection_group)
        
        #self.odom_sub = self.create_subscription(
        #    Odometry,
        #    '/fsds/testing_only/odom',
        #    self.processing_callback,
        #    10
        #)
        
        self.publisher_h = self.create_publisher(PointStamped, 'Stereo/h_cone_coords', 10)
        self.publisher = self.create_publisher(PointStamped, 'Stereo/cone_coords', 10)
        self.rviz_img = self.create_publisher(Image, 'Stereo/predicted_image', 10)
        
        self.process_timer = self.create_timer(
            0.01,
            self.processing_callback,
            callback_group = self.processing_group
        )


    def listener_callback(self, msg):
        try:
            # Ricezione immagine originale
            original_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            conf = 0.3
            # Ultralytics gestisce internamente la normalizzazione e il formato
            result = self.model.predict(original_frame, conf=conf, verbose=False)[0]
        
            all_detections = []
            boxes = result.boxes
            for box in boxes:
                score = box.conf[0].item()
                class_id = box.cls[0].item()
                # Coordinate [x1, y1, x2, y2]
                coords = box.xyxy[0].cpu().numpy()
                all_detections.append((score, class_id, coords))
                det2d = (msg.header, tuple(coords))
                self.queue.put(det2d)


            if not self.test:
                return

            #immagine da mostrare per vedere i coni:
            display_frame = original_frame #cv2.cvtColor(original_frame, cv2.COLOR_RGB2BGR)

            for i, det in enumerate(all_detections):
                score, class_id, coords = det
                x1, y1, x2, y2 = map(round, coords)

                #id_cone_map = {0: 'large_orange_cone', 1: 'orange_cone', 2: 'yellow_cone', 3: 'unknown_cone', 4: 'blue_cone'}

                # 6. Disegno su display_frame
                color = {
                    0: (0,0,255), #large orange
                    1: (0,165,255), #orange
                    2: (0,255,255), #yellow
                    3: (0,0,0),  #unknown
                    4: (255,0,0) #blue
                }
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color[class_id], 1)

                #text = f"{class_id}"
                #cv2.putText(display_frame, text, (x1, y1 - 10), 
                #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            #'''
            cv2.imwrite(f'output_{conf}.png', display_frame)
            '''
            with open("coords.txt", "w") as file:
                for i, det in enumerate(all_detections):
                    file.write(f"ID: {i}, score: {det[0]}, class: {int(det[1])}, coords: {det[2].tolist()}\n")
        
            # 7. Visualizzazione
            #cv2.imshow("openvino (640x640)", display_frame)
            #cv2.waitKey(10000)

            '''
            self.get_logger().warn("exiting as test image was produced")
            self.destroy_node()
            self.rviz_img.publish(self.bridge.cv2.to_imgmsg(display_frame, encoding="rgb8"))

        except Exception as e:
            self.get_logger().error(f'Errore callback: {str(e)}')
        
        try:
            for det in all_detections:
                self.queue.put_nowait((msg.header, det))
        except queue.Full:
            self.get_logger().warn("Processing thread is too slow! Dropping frame data.")


    def processing_callback(self):
        if self.queue.empty():
            return

        header, det = self.data_queue.get() 
        x1, _, x2, y2 = det

        x = (x2+x1)/2.
        y = y2

        x_c, y_c, z_c = (x-self.cx)/self.fx, (y-self.cy)/self.fy, 1.

        try:
            camera_frame = header.frame_id  
            transform = self.tf_buffer.lookup_transform(
                'map', 
                camera_frame, 
                header.stamp,
                rclpy.duration.Duration(seconds=0.1)  
            )
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(f"TF lookup failed: {e}")
            return
        
        cam_origin_local = PointStamped()
        cam_origin_local.header = header
        cam_origin_local.point.x = 0.0
        cam_origin_local.point.y = 0.0
        cam_origin_local.point.z = 0.0

        ray_end_local = PointStamped()
        ray_end_local.header = header
        ray_end_local.point.x = x_c
        ray_end_local.point.y = y_c
        ray_end_local.point.z = z_c

        cam_origin_world = tf2_geometry_msgs.do_transform_point(cam_origin_local, transform)
        ray_end_world = tf2_geometry_msgs.do_transform_point(ray_end_local, transform)

        #coordinates in world frame
        c_x, c_y, c_z = cam_origin_world.point.x, cam_origin_world.point.y, cam_origin_world.point.z
        r_x = ray_end_world.point.x - c_x
        r_y = ray_end_world.point.y - c_y
        r_z = ray_end_world.point.z - c_z

        if rz >= 0: return
        t = -c_z / rz

        x_global = c_x + (t * r_x)
        y_global = c_y + (t * r_y)
    
        p_w = (x2-x1)
        heuristic_factor = (1+p_w/(2*self.fx))

        msg_h, msg = PointStamped(), PointStamped()
        msg_h.header.stamp = header.stamp
        msg_h.header.frame_id = "map"
        msg_h.x, msg_h.y, msg_h.z = x_global*heuristic_factor, y_global*heuristic_factor, 0.

        msg.header.stamp = header.stamp
        msg.header.frame_id = "map"
        msg.x, msg.y, msg.z = x_global, y_global, 0.


        self.publisher.publish(msg)
        self.publisher_h.publish(msg_h)

def main(args=None):
    rclpy.init(args=args)
    image_listener = ImageListener()

    executor = MultiThreadedExecutor()
    executor.add_node(image_listener)
    try:
        executor.spin()
    finally:
        image_listener.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
