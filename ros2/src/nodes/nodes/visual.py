import rclpy
from rclpy.node import Node
from fs_msgs.msg import Track # Double check your team's specific message package
from visualization_msgs.msg import Marker, MarkerArray
from rclpy.qos import QoSProfile, DurabilityPolicy, HistoryPolicy

from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from rclpy.time import Time
# 1. Create a QoS profile with "Transient Local" durability (this is ROS 2 latching)
qos_profile = QoSProfile(depth=1)
qos_profile.durability = DurabilityPolicy.TRANSIENT_LOCAL
qos_profile.history=HistoryPolicy.KEEP_LAST



class ConeVisualizer(Node):
    def __init__(self):
        super().__init__('cone_visualizer')
        self.subscription = self.create_subscription(Track, '/fsds/testing_only/track', self.track_callback, qos_profile)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.odom = self.create_subscription(
            Odometry,
            '/fsds/testing_only/odom',
            self.odom_callback,
            10)
        self.publisher = self.create_publisher(MarkerArray, '/track_markers', qos_profile)
        self.processed = False

    def odom_callback(self, msg: Odometry):
        t = TransformStamped()
        t.header.stamp = msg.header.stamp
        t.header.frame_id = 'fsds/map'
        t.child_frame_id = 'fsds/FSCar'

        # Set translation
        t.transform.translation.x = msg.pose.pose.position.x
        t.transform.translation.y = msg.pose.pose.position.y
        t.transform.translation.z = msg.pose.pose.position.z

        # Set rotation
        t.transform.rotation = msg.pose.pose.orientation

        self.tf_broadcaster.sendTransform(t)

    def track_callback(self, msg):
        if self.processed:
            return

        self.get_logger().info("Received track layout, converting cones to markers")
        marker_array = MarkerArray()
        marker_id = 0

        # Helper function to generate individual cylinder markers
        def create_cone_marker(cone, color, unique_id):
            marker = Marker()
            # FSDS ground truth track is published in the 'map' frame
            marker.header.frame_id = "fsds/map"
            marker.header.stamp = Time().to_msg()
            marker.ns = "track_cones"

            marker.id = unique_id
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            
            # Position (Note: fs_msgs/Cone has a 'location' geometry_msgs/Point object)
            marker.pose.position.x = float(cone.location.x)
            marker.pose.position.y = float(cone.location.y)
            marker.pose.position.z = 0.15 # Offset slightly up so they sit on the ground
            
            marker.pose.orientation.w = 1.0 # No rotation needed for cylinders
            
            # Cone Dimensions (Diameter: 0.2m, Height: 0.3m)
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.3
            
            # Colors [Red, Green, Blue, Alpha (Transparency)]
            marker.color.r, marker.color.g, marker.color.b, marker.color.a = color[0], color[1], color[2], 1.0

            return marker

        color = {
                2: (1.,0.,0.), #large orange
                3: (1.,0.65,0.), #orange
                4: (0.,0.,0.), #unknown 
                1: (1.,1.,0.), #yellow
                0: (0.,0.,1.) #blue
            }
        for cone in msg.track:
            marker_array.markers.append(create_cone_marker(cone, color[cone.color], marker_id))
            marker_id += 1

        # Publish the finalized marker array
        self.publisher.publish(marker_array)
        self.get_logger().info(f"Successfully published {marker_id} cones to /track_markers. Locking data in memory.")
        
        # Set flag to True so we don't recalculate this again
        self.processed = True


def main(args=None):
    rclpy.init(args=args)
    node = ConeVisualizer()
    try:
        rclpy.spin(node) # Keep node alive to serve the cached map to RViz
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()