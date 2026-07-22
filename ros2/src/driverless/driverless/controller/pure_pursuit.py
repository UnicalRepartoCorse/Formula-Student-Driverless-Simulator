import math
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path, Odometry
from fs_msgs.msg import ControlCommand
from fs_msgs.srv import Reset
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from visualization_msgs.msg import Marker

class PurePursuitNode(Node):
    """
    Optimized Pure Pursuit controller with linear interpolation and
    dynamic speed scaling based on steering angle.
    Aggressive version with high acceleration to release brakes.
    """
    def __init__(self):
        super().__init__('pure_pursuit_controller')

        # --- Parameters ---
        self.lookahead_distance = 2.5  # meters
        self.wheelbase = 1.58          # meters
        self.max_steering_angle = math.radians(24)  # radians
        self.min_speed = 0.7         # m/s minimum speed (reduced for slow driving)

        # State variables
        self.path = []
        self.global_path = []
        self.current_speed = 0.0
        self.car_x = 0.0
        self.car_y = 0.0
        self.car_yaw = 0.0

        # --- Subscribers & Publishers ---
        self.path_sub = self.create_subscription(
            Path,
            '/planning/trajectory', #FOR RRT PATH FOLLOWING
            #'/track/centerline', #FOR CENTERLINE PATH FOLLOWING
            self.path_callback,
            10
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            '/fsds/testing_only/odom',
            self.odom_callback,
            10
        )

        self.cmd_pub = self.create_publisher(ControlCommand, '/fsds/control_command', 10)

        # TF Broadcaster to publish dynamic transform of the car for RViz
        self.tf_broadcaster = TransformBroadcaster(self)

        # Publisher to show a 3D representation of the car in RViz
        self.car_viz_pub = self.create_publisher(Marker, '/viz/car_model', 10)

        # Control loop timer (20 Hz)
        self.timer = self.create_timer(0.05, self.control_loop)

        self.get_logger().info("Pure Pursuit Node (Aggressive Version) started.")

    def path_callback(self, msg):
        self.global_path = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        #self.get_logger().info(f"Received path with {len(self.global_path)} points.")

    def odom_callback(self, msg):
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        self.current_speed = math.hypot(vx, vy)
        
        self.car_x = msg.pose.pose.position.x
        self.car_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.car_yaw = math.atan2(siny_cosp, cosy_cosp)

        # Broadcast the car's TF relative to map so RViz can follow it
        t = TransformStamped()
        t.header.stamp = msg.header.stamp
        t.header.frame_id = 'fsds/map'
        t.child_frame_id = 'fsds/FSCar'
        t.transform.translation.x = self.car_x
        t.transform.translation.y = self.car_y
        t.transform.translation.z = msg.pose.pose.position.z
        t.transform.rotation = q
        self.tf_broadcaster.sendTransform(t)

        # Publish a 3D box representing the physical car footprint
        car_marker = Marker()
        car_marker.header.stamp = msg.header.stamp
        car_marker.header.frame_id = 'fsds/FSCar'
        car_marker.ns = 'car'
        car_marker.id = 0
        car_marker.type = Marker.CUBE
        car_marker.action = Marker.ADD
        car_marker.scale.x = 2.0  # Length (meters)
        car_marker.scale.y = 1.2  # Width (meters)
        car_marker.scale.z = 0.6  # Height (meters)
        car_marker.pose.position.x = 0.75  # Shift center forward (wheelbase center)
        car_marker.pose.position.y = 0.0
        car_marker.pose.position.z = 0.3  # Half height
        car_marker.pose.orientation.w = 1.0
        car_marker.color.r = 1.0  # Orange color
        car_marker.color.g = 0.3
        car_marker.color.b = 0.0
        car_marker.color.a = 0.8  # Semi-transparent
        self.car_viz_pub.publish(car_marker)

    def get_lookahead_point(self):
        if not self.path:
            return None

        closest_idx = -1
        min_dist = float('inf')
        for i, (px, py, _) in enumerate(self.path):
            if px > 0.0:
                dist = math.hypot(px, py)
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = i

        if closest_idx == -1:
            return self.path[-1]

        for i in range(closest_idx, len(self.path)):
            px, py, speed = self.path[i]
            dist = math.hypot(px, py)

            if dist >= self.lookahead_distance:
                if i > 0:
                    prev_px, prev_py, prev_speed = self.path[i-1]
                    prev_dist = math.hypot(prev_px, prev_py)
                    dist_diff = dist - prev_dist
                    if dist_diff > 0.001:
                        t = (self.lookahead_distance - prev_dist) / dist_diff
                        interp_px = prev_px + t * (px - prev_px)
                        interp_py = prev_py + t * (py - prev_py)
                        interp_speed = prev_speed + t * (speed - prev_speed)
                        return (interp_px, interp_py, interp_speed)
                return (px, py, speed)

        return self.path[-1]

    def control_loop(self):
        # Transform global path to local frame
        self.path = []
        for gx, gy in self.global_path:
            dx = gx - self.car_x
            dy = gy - self.car_y
            lx = dx * math.cos(self.car_yaw) + dy * math.sin(self.car_yaw)
            ly = -dx * math.sin(self.car_yaw) + dy * math.cos(self.car_yaw)
            self.path.append((lx, ly, 2.0))

        msg = ControlCommand()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "driverless_car"

        if not self.path:
            msg.throttle = 0.0
            msg.steering = 0.0
            msg.brake = 1.0
            self.cmd_pub.publish(msg)
            return

        target = self.get_lookahead_point()
        if target is None:
            msg.throttle = 0.0
            msg.steering = 0.0
            msg.brake = 1.0
            self.cmd_pub.publish(msg)
            return

        lx, ly, target_speed = target
        ld_squared = lx**2 + ly**2

        if ld_squared < 0.0001:
            steering_angle = 0.0
        else:
            steering_angle = math.atan2(2.0 * self.wheelbase * ly, ld_squared)

        steering_angle = max(-self.max_steering_angle, min(self.max_steering_angle, steering_angle))

        # Speed Management
        speed_factor = 1.0 - (abs(steering_angle) / self.max_steering_angle)

        # STRICT SPEED LIMIT FOR DEBUGGING
        # Ignore target_speed from RRT completely to guarantee slow movement
        target_v = 6
        final_speed = self.min_speed + (target_v - self.min_speed) * speed_factor

        # Simple proportional speed controller
        speed_error = final_speed - self.current_speed

        msg.steering = -float(steering_angle / self.max_steering_angle)

        if speed_error > 0:
            msg.throttle = float(max(0.0, min(1.0, 0.4 * speed_error)))
            msg.brake = 0.0
        else:
            msg.throttle = 0.0
            msg.brake = float(max(0.0, min(1.0, 0.2 * (-speed_error))))

        self.cmd_pub.publish(msg)

        if not hasattr(self, '_ctrl_log_count'):
            self._ctrl_log_count = 0
        self._ctrl_log_count += 1
        if self._ctrl_log_count % 20 == 0:
            self.get_logger().info(f"Ctrl: throttle={msg.throttle:.2f}, steering={msg.steering:.2f}, brake={msg.brake:.2f}, speed_err={speed_error:.2f}")

    def reset_simulator(self):
        #self.get_logger().info("Sending reset request to /fsds/reset...")
        #import subprocess
        #try:
        #    # We call the service via subprocess to ensure it runs reliably
        #    # even during KeyboardInterrupt / node shutdown sequence.
        #    subprocess.run(
        #        ["ros2", "service", "call", "/fsds/reset", "fs_msgs/srv/Reset", "{wait_on_last_task: false}"],
        #        stdout=subprocess.DEVNULL,
        #        stderr=subprocess.DEVNULL,
        #        timeout=2.0
        #    )
        #    self.get_logger().info("Reset request completed.")
        #except Exception as e:
        #    self.get_logger().error(f"Failed to call reset service via subprocess: {e}")
        pass
def main(args=None):
    rclpy.init(args=args)
    node = PurePursuitNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.reset_simulator()
        except Exception as e:
            node.get_logger().error(f"Failed to reset simulator: {e}")
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
