# RRT* Path Planning Architecture for Driverless Simulator (FSDS)

## Objective
Implement a Kinematic RRT* path planning algorithm to navigate through a track defined by cones. The implementation is modular, testable, and fully integrated with the ROS 2 stack for the Formula Student Driverless Simulator.

## Key Files & Context
- **Package:** `driverless` (ROS 2 package)
- **Location:** `ros2/src/driverless/driverless/path_planning/`
- **Interfaces:** `fs_msgs/msg/Track` (Input Cones), `nav_msgs/msg/Path` (Input Centerline & Output Trajectory), `nav_msgs/msg/Odometry` (Vehicle Pose).

## Architecture Details

The system is divided into two main layers: the Core Algorithm and the ROS 2 Wrapper.

### 1. Core Algorithm Layer (Pure Python)
This layer contains the mathematical and logical implementation of the Kinematic RRT*, agnostic to ROS 2.
- **`rrt_star.py`**: Contains the main `RRTStar` class.
    - **Sampling:** Generates random points in the configuration space `(x, y, theta)`. Supports centerline-guided sampling within a specified radius (`sample_radius_centerline`) to accelerate search and stay within boundaries.
    - **Steering (Kinematic):** Integrates a discrete bicycle model to steer from the nearest node toward the sampled point. The required heading change is mapped to the steering angle, which is explicitly clipped to `max_steering_angle` (e.g., ±24 degrees) rather than being rejected. This ensures all generated trajectories are kinematically feasible.
    - **Cost/Rewiring (RRT*):** Calculates trajectory costs including a centerline deviation penalty and a highly optimized heading penalty (based on the squared kinematic steering ratio without trigonometric functions) to favor straighter, smoother paths. Connects new nodes to the best available parents and performs rewiring using an iterative BFS approach (`deque`) to rapidly propagate cost updates down the tree, avoiding Python's recursion limits.
- **`collision_checker.py`** (in `driverless.utils`): A dedicated module for collision detection.
    - Checks if the simulated trajectory collides with any cone within a threshold safety radius.

### 2. ROS 2 Wrapper Layer
This layer interfaces the core logic with the ROS 2 workspace.
- **`rrt_node.py`**: The `rclpy` node.
    - **Parameters:**
        - `collision_strategy` (string): Strategy for collision checking (default: `'radial'`).
        - `max_steering_angle` (float): Maximum wheel angle in radians.
        - `wheelbase` (float): Distance between axles in meters.
        - `step_size` (float): Expansion step size in meters.
        - `sample_radius_centerline` (float): Centerline search radius.
        - `max_iter` (int): Maximum planner iterations.
    - **Subscribers:**
        - `/fsds/testing_only/track` (`fs_msgs/msg/Track`): Cone map.
        - `/track/centerline` (`nav_msgs/msg/Path`): Optional centerline.
        - `/fsds/testing_only/odom` (`nav_msgs/msg/Odometry`): Vehicle pose.
    - **Publishers:**
        - `/planning/trajectory` (`nav_msgs/msg/Path`): Trajectory for the Pure Pursuit controller.
        - `/planning/viz` (`visualization_msgs/msg/MarkerArray`): Tree, nodes, sample points, goal line, and path for RViz.
    - **Execution Logic (10 Hz):**
        - Tracks seen cones. Only triggers a replan when `NEW_CONE_THRESHOLD` (3) new cones are detected on both sides, or if no path exists.
        - Stitches path: Starts the RRT* search from the `last_point` node (default: 3rd from the end) of the previous path to ensure smooth kinematic continuity between replans.
        - Trims path: Discards waypoints more than 3 meters behind the car.

## Verification & Testing
- **RViz Visualization:** Run `rviz2` and visualize markers under `/planning/viz` to inspect the tree growth, sample points, local goal line, and the smoothed final path.
- **Integration Test:** Verify that the Pure Pursuit controller executes the published trajectory smoothly in the FSDS environment.
