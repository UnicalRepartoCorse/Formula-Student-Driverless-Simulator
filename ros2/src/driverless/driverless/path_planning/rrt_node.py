import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from fs_msgs.msg import Track
from geometry_msgs.msg import Point, PoseStamped
from visualization_msgs.msg import MarkerArray, Marker
from builtin_interfaces.msg import Duration

from nav_msgs.msg import Path, Odometry
from driverless.utils.collision_checker import CollisionChecker
from driverless.utils.utils import normalize_angle
from .rrt_star import RRTStar
from .rrt_star import Node as RRTNodeState

# ENU FRAME

class RRTNode(Node):
    """
    ROS 2 node that wraps the RRT* Path Planner.
    Subscribes to cones and centerline, publishes the trajectory for the Pure Pursuit controller.
    """

    def __init__(self):
        super().__init__('rrt_path_planner')

        # --- Parameters ---
        # passo 1.58m e carreggiata anterio 1.27m
        # angolo sterzo 120 gradi e rapporto 5:1 -> angolo ruota 24 gradi
        self.declare_parameter('collision_strategy', 'radial')
        self.declare_parameter('max_steering_angle', math.radians(24)) # rad
        self.declare_parameter('wheelbase', 1.58) # m
        self.declare_parameter('step_size', 0.2) # m
        self.declare_parameter('sample_radius_centerline', 1.4)
        self.declare_parameter('max_iter', 700)
        self.declare_parameter('num_trees', 1)
        self.declare_parameter('last_point', 3)
        self.declare_parameter('centerline_topic', '/track/centerline')
        self.declare_parameter('cones_topic', '/fsds/testing_only/track')

        # Read parameters
        collision_strategy = self.get_parameter('collision_strategy').value
        centerline_topic = self.get_parameter('centerline_topic').value
        cones_topic = self.get_parameter('cones_topic').value

        # State
        self.centerline = []
        self.car_x = 0.0
        self.car_y = 0.0
        self.car_yaw = 0.0

        # --- Cone-triggered replanning state ---
        # Coni visti dall'inizio, usati per rilevare nuovi coni
        self.seen_blue_keys:   set = set()
        self.seen_yellow_keys: set = set()
        # Numero di coni visti all'ultimo plan
        self.n_blue_at_last_plan:   int = 0
        self.n_yellow_at_last_plan: int = 0
        # Soglia: quanti NUOVI coni per lato triggherano un replan
        self.NEW_CONE_THRESHOLD: int = 3
        # Ultimo punto del path precedente in frame GLOBALE (gx, gy, gtheta)
        self.last_goal_global = None
        # Ultima goal line calcolata in coordinate GLOBALI ((gx1, gy1), (gx2, gy2))
        self.last_goal_line_global = None
        # Path completo pubblicato in frame globale (lista di (gx, gy))
        self.published_path_global = []
        # Lista di stati globali (gx, gy, gtheta) del path precedente
        self.last_path_global_states = []

        # --- Subscribers & Publishers ---
        cone_qos = QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
        self.cone_sub = self.create_subscription(
            Track,
            cones_topic,
            self.cones_callback,
            cone_qos
        )

        self.centerline_sub = self.create_subscription(
            Path,
            centerline_topic,
            self.centerline_callback,
            10
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            '/fsds/testing_only/odom',
            self.odom_callback,
            10
        )

        self.trajectory_pub = self.create_publisher(
            Path,
            '/planning/trajectory',
            10
        )

        # Publisher for RViz visualization
        self.viz_pub = self.create_publisher(MarkerArray, '/planning/viz', 10)

        # Publishers for boundary paths
        self.blue_boundary_pub = self.create_publisher(
            Path,
            '/planning/blue_boundary',
            10
        )
        self.yellow_boundary_pub = self.create_publisher(
            Path,
            '/planning/yellow_boundary',
            10
        )

        self.last_track_msg = None
        self.timer = self.create_timer(0.1, self.plan_timer_callback) # 10 Hz

        # Initialize the collision checker
        self.collision_checker = CollisionChecker(strategy=collision_strategy, cone_radius=0.7)

        self.get_logger().info(f"RRT* Node Initialized. Subscribed to {cones_topic} with strategy {collision_strategy}")

    def centerline_callback(self, msg: Path):
        """
        Callback triggered when a new centerline is received.
        """
        self.centerline = [RRTNodeState(p.pose.position.x, p.pose.position.y, 0) for p in msg.poses]
        #self.get_logger().info(f"Received centerline with {len(self.centerline)} points.")

    def odom_callback(self, msg: Odometry):
        self.car_x = msg.pose.pose.position.x
        self.car_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.car_yaw = math.atan2(siny_cosp, cosy_cosp)

        if not hasattr(self, '_odom_log_count'):
            self._odom_log_count = 0

    def cones_callback(self, msg: Track):
        self.last_track_msg = msg

    def plan_timer_callback(self): #TODO rivedere con coni da perception
        if self.last_track_msg is None:
            return

        msg = self.last_track_msg
        # 1. Extract cone coordinates from msg and transform to local frame
        global_blue_cones = []
        global_yellow_cones = []
        global_orange_cones = []

        local_blue_cones = []
        local_yellow_cones = []
        local_orange_cones = []

        for c in msg.track:
            gx, gy = c.location.x, c.location.y
            dist = math.hypot(gx - self.car_x, gy - self.car_y)
            if dist < 10.0:
                dx = gx - self.car_x
                dy = gy - self.car_y
                lx = dx * math.cos(self.car_yaw) + dy * math.sin(self.car_yaw)
                ly = -dx * math.sin(self.car_yaw) + dy * math.cos(self.car_yaw)
                pos_local = (lx, ly)
                pos_global = (gx, gy)

                if c.color == 0: # BLUE
                    local_blue_cones.append(pos_local)
                    global_blue_cones.append(pos_global)
                elif c.color == 1: # YELLOW
                    local_yellow_cones.append(pos_local)
                    global_yellow_cones.append(pos_global)
                elif c.color in (2, 3): # ORANGE
                    local_orange_cones.append(pos_local)
                    global_orange_cones.append(pos_global)

        # Esegui l'update dei coni e ricevi i contorni calcolati in frame locale
        blue_boundary_local, yellow_boundary_local = self.collision_checker.update_cones(
            local_blue_cones, local_yellow_cones, local_orange_cones
        )

        # Converti i contorni in coordinate globali per la visualizzazione su RViz
        if blue_boundary_local is not None and len(blue_boundary_local) > 0:
            self.last_blue_boundary = [self.to_global(p[0], p[1]) for p in blue_boundary_local]
        else:
            self.last_blue_boundary = []

        if yellow_boundary_local is not None and len(yellow_boundary_local) > 0:
            self.last_yellow_boundary = [self.to_global(p[0], p[1]) for p in yellow_boundary_local]
        else:
            self.last_yellow_boundary = []

        # Pubblica i contorni su topic dedicati
        self.publish_boundary_paths()

        # Collect all coordinates for bounds calculation
        all_local_cones = local_blue_cones + local_yellow_cones + local_orange_cones
        if not all_local_cones:
            return

        lx_coords = [c[0] for c in all_local_cones]
        ly_coords = [c[1] for c in all_local_cones]

        # 2. Aggiorna i coni visti (con chiave arrotondata per robustezza float)
        def cone_key(gx, gy): return (round(gx, 1), round(gy, 1))
        for gx, gy in global_blue_cones:   self.seen_blue_keys.add(cone_key(gx, gy))
        for gx, gy in global_yellow_cones: self.seen_yellow_keys.add(cone_key(gx, gy))

        new_blue   = len(self.seen_blue_keys)   - self.n_blue_at_last_plan
        new_yellow = len(self.seen_yellow_keys) - self.n_yellow_at_last_plan

        # Replan solo se ci sono abbastanza nuovi coni su ENTRAMBI i lati
        # oppure se non abbiamo ancora nessun path
        should_replan = (
            self.last_goal_global is None or
            (new_blue >= self.NEW_CONE_THRESHOLD and
             new_yellow >= self.NEW_CONE_THRESHOLD)
        )

        if not should_replan:
            # Nessun replan: ripubblica il path già calcolato e aggiorna viz
            self._trim_published_path()
            self.publish_path(self.published_path_global)
            self.publish_viz(self.published_path_global, [], self.last_goal_line_global,
                             global_blue_cones, global_yellow_cones, global_orange_cones, None)
            return

        # 3. Centerline in frame locale
        local_centerline = []
        if self.centerline:
            for node in self.centerline:
                dx = node.x - self.car_x
                dy = node.y - self.car_y
                lx =  dx * math.cos(self.car_yaw) + dy * math.sin(self.car_yaw)
                ly = -dx * math.sin(self.car_yaw) + dy * math.cos(self.car_yaw)
                local_centerline.append(RRTNodeState(lx, ly,0))

        # 4. Goal line in frame locale
        goal_line = None

        if local_centerline and len(local_centerline) > 1:
            p1 = local_centerline[-2]
            p2 = local_centerline[-1]
            dx = p2.x - p1.x
            dy = p2.y - p1.y
            length = math.hypot(dx, dy)
            if length > 0:
                nx = -dy / length * 2.0
                ny =  dx / length * 2.0
                goal_line = ((p2.x + nx, p2.y + ny), (p2.x - nx, p2.y - ny))

        if goal_line is None and local_blue_cones and local_yellow_cones:
            last_blue   = max(local_blue_cones,   key=lambda c: c[0])
            last_yellow = max(local_yellow_cones, key=lambda c: c[0])
            goal_line = (last_blue, last_yellow)

        if goal_line is None and all_local_cones:
            furthest_x = max(lx_coords)
            goal_line = ((furthest_x, -2.0), (furthest_x, 2.0))

        if goal_line is None:
            self.get_logger().warn("Goal None - skipping replan")
            return

        # Salva la goal_line in coordinate globali per visualizzarla in modo persistente
        self.last_goal_line_global = (
            self.to_global(goal_line[0][0], goal_line[0][1]),
            self.to_global(goal_line[1][0], goal_line[1][1])
        )

        # 5. Start del nuovo RRT: prendiamo il 4° punto dalla fine del path precedente (se disponibile)
        #    per garantire una transizione liscia, altrimenti partiamo dall'origine (posizione auto).

        last_point = self.get_parameter('last_point').value
        if self.last_path_global_states and len(self.last_path_global_states) >=last_point:
            start_state_global = self.last_path_global_states[-last_point]
            lgx, lgy, lgtheta = start_state_global
            dx  = lgx - self.car_x
            dy  = lgy - self.car_y
            slx =  dx * math.cos(self.car_yaw) + dy * math.sin(self.car_yaw)
            sly = -dx * math.sin(self.car_yaw) + dy * math.cos(self.car_yaw)
            stheta = lgtheta - self.car_yaw
            stheta = normalize_angle(stheta)
            rrt_start = (slx, sly, stheta)
            overlap_count = last_point  # Rimuoveremo gli ultimi 4 punti per agganciarci al 5°
        elif self.last_goal_global is not None:
            lgx, lgy, lgtheta = self.last_goal_global
            dx  = lgx - self.car_x
            dy  = lgy - self.car_y
            slx =  dx * math.cos(self.car_yaw) + dy * math.sin(self.car_yaw)
            sly = -dx * math.sin(self.car_yaw) + dy * math.cos(self.car_yaw)
            stheta = lgtheta - self.car_yaw
            stheta = (stheta + math.pi) % (2 * math.pi) - math.pi
            rrt_start = (slx, sly, stheta)
            overlap_count = 0
        else:
            rrt_start = (0.0, 0.0, 0.0)
            overlap_count = 0


        # Calcolo dei target per il campionamento (target-biased)
        rrt_targets = []
        cone_obstacle_size = 1.1  # raggio/dimensione ostacolo del cono
        for lx, ly in local_blue_cones + local_yellow_cones:
            if lx > 1.5:  # considera solo coni davanti al veicolo
                rrt_targets.append((lx, ly, cone_obstacle_size))

        trees = []
        for i in range(self.get_parameter('num_trees').value):
            # 7. Run RRT* dal punto di estensione
            rrt = RRTStar(
                start=rrt_start,
                goal_line=goal_line,
                collision_checker=self.collision_checker,
                max_steering_angle=self.get_parameter('max_steering_angle').value,
                wheelbase=self.get_parameter('wheelbase').value,
                step_size=self.get_parameter('step_size').value,
                max_iter=self.get_parameter('max_iter').value,
                sample_radius_centerline=self.get_parameter('sample_radius_centerline').value,
                centerline=local_centerline if local_centerline else None,
                rrt_targets=None # rrt_targets
            )
            res = rrt.plan()
            if res is not None:
                new_path_local, path_cost = res
                trees.append((new_path_local, path_cost))

        if trees:
            new_path_local, path_cost = min(trees, key=lambda x: x[1])
        else:
            new_path_local = None

        if new_path_local is not None:
            # Converti nuovo path in frame globale
            new_path_global = [self.to_global(p[0], p[1]) for p in new_path_local]

            # Stitch: trim path precedente (punti dietro la macchina)
            self._trim_published_path()

            # Rimuovi gli ultimi 3 elementi sovrapposti per collegare correttamente il nuovo percorso
            if 0 < overlap_count <= len(self.published_path_global):
                self.published_path_global = self.published_path_global[:-overlap_count]

            self.published_path_global.extend(new_path_global)

            # Salva l'ultimo punto in globale per fallback
            last_local = new_path_local[-1]
            lgx, lgy   = self.to_global(last_local[0], last_local[1])
            lgtheta    = last_local[2] + self.car_yaw
            lgtheta    = normalize_angle(lgtheta)
            self.last_goal_global = (lgx, lgy, lgtheta)

            # Salva tutti i nuovi stati globali per il prossimo replan
            self.last_path_global_states = []
            for p in new_path_local:
                gx, gy = self.to_global(p[0], p[1])
                gtheta = p[2] + self.car_yaw
                gtheta = normalize_angle(gtheta)
                self.last_path_global_states.append((gx, gy, gtheta))

            # Aggiorna soglia coni
            self.n_blue_at_last_plan   = len(self.seen_blue_keys)
            self.n_yellow_at_last_plan = len(self.seen_yellow_keys)

        # 8. Pubblica e visualizza
        self.publish_path(self.published_path_global)
        self.publish_viz(self.published_path_global, rrt.node_list, self.last_goal_line_global,
                         global_blue_cones, global_yellow_cones, global_orange_cones, rrt.sampled_points)


    def _trim_published_path(self):
        """Rimuove i waypoint già percorsi (più di 3 m dietro la macchina)."""
        if not self.published_path_global:
            return
        cos_y = math.cos(self.car_yaw)
        sin_y = math.sin(self.car_yaw)
        trimmed = []
        for gx, gy in self.published_path_global:
            dx = gx - self.car_x
            dy = gy - self.car_y
            lx = dx * cos_y + dy * sin_y  # coordinata longitudinale locale
            if lx > -3.0:  # tieni tutto ciò che non è più di 3 m dietro
                trimmed.append((gx, gy))
        self.published_path_global = trimmed

    def to_global(self, lx, ly):
        gx = self.car_x + lx * math.cos(self.car_yaw) - ly * math.sin(self.car_yaw)
        gy = self.car_y + lx * math.sin(self.car_yaw) + ly * math.cos(self.car_yaw)
        return gx, gy

    def publish_viz(self, path, nodes, goal, blue_cones, yellow_cones, orange_cones, samples=None):
        """
        Publishes markers for the RRT* tree, the final path, the goal, and the centerline.
        """
        marker_array = MarkerArray()
        now = self.get_clock().now().to_msg()
        lifetime_msg = Duration(sec=400, nanosec=50) # Aumentato a 1.5s per eliminare lo sfarfallio

        # 1. Tree Marker (Line List)
        tree_marker = Marker()
        tree_marker.header.frame_id = "fsds/map"
        tree_marker.header.stamp = now
        tree_marker.ns = "rrt_tree"
        tree_marker.id = 0
        tree_marker.type = Marker.LINE_LIST
        tree_marker.action = Marker.ADD
        tree_marker.pose.orientation.w = 1.0
        tree_marker.scale.x = 0.02 # Line width
        tree_marker.color.r = 0.6
        tree_marker.color.g = 0.1
        tree_marker.color.b = 0.8
        tree_marker.color.a = 0.6 # Semi-transparent purple
        tree_marker.lifetime = lifetime_msg

        for node in nodes:
            if node.parent is not None:
                gx1, gy1 = self.to_global(node.parent.x, node.parent.y)
                gx2, gy2 = self.to_global(node.x, node.y)
                p1 = Point(x=float(gx1), y=float(gy1), z=0.0)
                p2 = Point(x=float(gx2), y=float(gy2), z=0.0)
                tree_marker.points.append(p1)
                tree_marker.points.append(p2)

        marker_array.markers.append(tree_marker)

        # 1b. Tree Nodes Marker (Points, light blue)
        if nodes:
            tree_nodes_marker = Marker()
            tree_nodes_marker.header.frame_id = "fsds/map"
            tree_nodes_marker.header.stamp = now
            tree_nodes_marker.ns = "rrt_tree_nodes"
            tree_nodes_marker.id = 15
            tree_nodes_marker.type = Marker.POINTS
            tree_nodes_marker.action = Marker.ADD
            tree_nodes_marker.pose.orientation.w = 1.0
            tree_nodes_marker.scale.x = 0.12  # Aumentato da 0.05 a 0.12 per visibilità
            tree_nodes_marker.scale.y = 0.12
            tree_nodes_marker.color.r = 0.6
            tree_nodes_marker.color.g = 0.1
            tree_nodes_marker.color.b = 0.8  # Purple
            tree_nodes_marker.color.a = 0.2
            tree_nodes_marker.lifetime = lifetime_msg

            for node in nodes:
                gx, gy = self.to_global(node.x, node.y)
                tree_nodes_marker.points.append(Point(x=float(gx), y=float(gy), z=0.0))

            marker_array.markers.append(tree_nodes_marker)

        # 1c. Sampled Points Marker (Points, semi-transparent orange)
        if samples:
            samples_marker = Marker()
            samples_marker.header.frame_id = "fsds/map"
            samples_marker.header.stamp = now
            samples_marker.ns = "rrt_samples"
            samples_marker.id = 20
            samples_marker.type = Marker.POINTS
            samples_marker.action = Marker.ADD
            samples_marker.pose.orientation.w = 1.0
            samples_marker.scale.x = 0.15  # Aumentato da 0.03 a 0.08 per visibilità
            samples_marker.scale.y = 0.08
            samples_marker.color.r = 1.0
            samples_marker.color.g = 0.4
            samples_marker.color.b = 0.0  # Orange
            samples_marker.color.a = 0.2  # Semi-transparent to avoid clutter
            samples_marker.lifetime = lifetime_msg

            for sx, sy in samples:
                gx, gy = self.to_global(sx, sy)
                samples_marker.points.append(Point(x=float(gx), y=float(gy), z=0.0))

            marker_array.markers.append(samples_marker)

        # 2. Path Marker (Line Strip)
        if path:
            path_marker = Marker()
            path_marker.header.frame_id = "fsds/map"
            path_marker.header.stamp = now
            path_marker.ns = "rrt_path"
            path_marker.id = 1
            path_marker.type = Marker.LINE_STRIP
            path_marker.action = Marker.ADD
            path_marker.pose.orientation.w = 1.0
            path_marker.scale.x = 0.1 # Line width
            path_marker.color.r = 0.0
            path_marker.color.g = 1.0
            path_marker.color.b = 0.0
            path_marker.color.a = 1.0
            path_marker.lifetime = lifetime_msg

            for p in path:
                path_marker.points.append(Point(x=float(p[0]), y=float(p[1]), z=0.0))

            marker_array.markers.append(path_marker)

            # 2b. Path Points Marker (Sphere List)
            path_points_marker = Marker()
            path_points_marker.header.frame_id = "fsds/map"
            path_points_marker.header.stamp = now
            path_points_marker.ns = "rrt_path_points"
            path_points_marker.id = 10
            path_points_marker.type = Marker.SPHERE_LIST
            path_points_marker.action = Marker.ADD
            path_points_marker.pose.orientation.w = 1.0
            path_points_marker.scale.x = 0.15  # Sphere diameter X
            path_points_marker.scale.y = 0.15  # Sphere diameter Y
            path_points_marker.scale.z = 0.15  # Sphere diameter Z
            path_points_marker.color.r = 1.0
            path_points_marker.color.g = 0.0
            path_points_marker.color.b = 0.0  # Red
            path_points_marker.color.a = 1.0
            path_points_marker.lifetime = lifetime_msg

            for p in path:
                path_points_marker.points.append(Point(x=float(p[0]), y=float(p[1]), z=0.0))

            marker_array.markers.append(path_points_marker)

        # 3. Goal Marker (solo durante replan, quando goal non è None)
        goal_marker = Marker()
        goal_marker.header.frame_id = "fsds/map"
        goal_marker.header.stamp = now
        goal_marker.ns = "rrt_goal"
        goal_marker.id = 2
        if goal is not None:
            goal_marker.type = Marker.LINE_STRIP
            goal_marker.action = Marker.ADD
            goal_marker.pose.orientation.w = 1.0
            goal_marker.scale.x = 0.1
            goal_marker.color.r = 1.0
            goal_marker.color.g = 0.0
            goal_marker.color.b = 1.0  # Magenta/Fuchsia
            goal_marker.color.a = 1.0
            goal_marker.lifetime = lifetime_msg
            goal_marker.points.append(Point(x=float(goal[0][0]), y=float(goal[0][1]), z=0.0))
            goal_marker.points.append(Point(x=float(goal[1][0]), y=float(goal[1][1]), z=0.0))
        else:
            goal_marker.action = Marker.DELETE
        marker_array.markers.append(goal_marker)

        # 4. Centerline Marker (Line Strip, Red)
        if self.centerline:
            cl_marker = Marker()
            cl_marker.header.frame_id = "fsds/map"
            cl_marker.header.stamp = now
            cl_marker.ns = "centerline_ref"
            cl_marker.id = 3
            cl_marker.type = Marker.LINE_STRIP
            cl_marker.action = Marker.ADD
            cl_marker.pose.orientation.w = 1.0
            cl_marker.scale.x = 0.05 # Slightly thinner than the RRT path
            cl_marker.color.r = 1.0
            cl_marker.color.g = 0.0
            cl_marker.color.b = 0.0
            cl_marker.color.a = 0.8 # Semi-transparent red
            cl_marker.lifetime = lifetime_msg

            for p in self.centerline:
                cl_marker.points.append(Point(x=float(p.x), y=float(p.y), z=0.0))

            marker_array.markers.append(cl_marker)

        # 4b. Blue Boundary Marker (Line Strip, Blue)
        if hasattr(self, 'last_blue_boundary') and self.last_blue_boundary:
            blue_b_marker = Marker()
            blue_b_marker.header.frame_id = "fsds/map"
            blue_b_marker.header.stamp = now
            blue_b_marker.ns = "blue_boundary"
            blue_b_marker.id = 50
            blue_b_marker.type = Marker.LINE_STRIP
            blue_b_marker.action = Marker.ADD
            blue_b_marker.pose.orientation.w = 1.0
            blue_b_marker.scale.x = 0.05
            blue_b_marker.color.r = 0.0
            blue_b_marker.color.g = 0.4
            blue_b_marker.color.b = 1.0 # Nice bright blue
            blue_b_marker.color.a = 0.8
            blue_b_marker.lifetime = lifetime_msg

            for p in self.last_blue_boundary:
                blue_b_marker.points.append(Point(x=float(p[0]), y=float(p[1]), z=0.0))

            marker_array.markers.append(blue_b_marker)

        # 4c. Yellow Boundary Marker (Line Strip, Yellow)
        if hasattr(self, 'last_yellow_boundary') and self.last_yellow_boundary:
            yellow_b_marker = Marker()
            yellow_b_marker.header.frame_id = "fsds/map"
            yellow_b_marker.header.stamp = now
            yellow_b_marker.ns = "yellow_boundary"
            yellow_b_marker.id = 51
            yellow_b_marker.type = Marker.LINE_STRIP
            yellow_b_marker.action = Marker.ADD
            yellow_b_marker.pose.orientation.w = 1.0
            yellow_b_marker.scale.x = 0.05
            yellow_b_marker.color.r = 1.0
            yellow_b_marker.color.g = 0.8
            yellow_b_marker.color.b = 0.0 # Nice bright yellow
            yellow_b_marker.color.a = 0.8
            yellow_b_marker.lifetime = lifetime_msg

            for p in self.last_yellow_boundary:
                yellow_b_marker.points.append(Point(x=float(p[0]), y=float(p[1]), z=0.0))

            marker_array.markers.append(yellow_b_marker)

        # 5. Cones Markers (CUBE_LIST for efficiency)
        def add_cones_marker(cones, marker_id, r, g, b):
            if not cones:
                return
            m = Marker()
            m.header.frame_id = "fsds/map"
            m.header.stamp = now
            m.ns = "rrt_cones"
            m.id = marker_id
            m.type = Marker.CUBE_LIST
            m.action = Marker.ADD
            m.pose.orientation.w = 1.0
            m.scale.x = 0.2
            m.scale.y = 0.2
            m.scale.z = 0.3
            m.color.r = r
            m.color.g = g
            m.color.b = b
            m.color.a = 1.0
            m.lifetime = lifetime_msg
            for cx, cy in cones:
                m.points.append(Point(x=float(cx), y=float(cy), z=0.15))
            marker_array.markers.append(m)

        add_cones_marker(blue_cones, 4, 0.0, 0.0, 1.0) # Blue
        add_cones_marker(yellow_cones, 5, 1.0, 1.0, 0.0) # Yellow
        add_cones_marker(orange_cones, 6, 1.0, 0.5, 0.0) # Orange

        self.viz_pub.publish(marker_array)

    def publish_path(self, path):
        """
        Convert a list of (x, y, theta) states into a Path and publish.
        """
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        # Ensure this matches the frame of your cones (usually 'base_footprint' or 'map')
        msg.header.frame_id = 'fsds/map'

        for state in path:
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x = float(state[0])
            pose.pose.position.y = float(state[1])
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)

        self.trajectory_pub.publish(msg)

    def publish_boundary_paths(self):
        """
        Publishes the blue and yellow boundaries as Path messages.
        """
        now = self.get_clock().now().to_msg()

        blue_msg = Path()
        blue_msg.header.stamp = now
        blue_msg.header.frame_id = 'fsds/map'
        if hasattr(self, 'last_blue_boundary') and self.last_blue_boundary:
            for p in self.last_blue_boundary:
                pose = PoseStamped()
                pose.header = blue_msg.header
                pose.pose.position.x = float(p[0])
                pose.pose.position.y = float(p[1])
                pose.pose.position.z = 0.0
                pose.pose.orientation.w = 1.0
                blue_msg.poses.append(pose)
        self.blue_boundary_pub.publish(blue_msg)

        yellow_msg = Path()
        yellow_msg.header.stamp = now
        yellow_msg.header.frame_id = 'fsds/map'
        if hasattr(self, 'last_yellow_boundary') and self.last_yellow_boundary:
            for p in self.last_yellow_boundary:
                pose = PoseStamped()
                pose.header = yellow_msg.header
                pose.pose.position.x = float(p[0])
                pose.pose.position.y = float(p[1])
                pose.pose.position.z = 0.0
                pose.pose.orientation.w = 1.0
                yellow_msg.poses.append(pose)
        self.yellow_boundary_pub.publish(yellow_msg)

def main(args=None):
    rclpy.init(args=args)
    node = RRTNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Publish empty path on shutdown to clear the path line in RViz
        try:
            empty_path = Path()
            empty_path.header.stamp = node.get_clock().now().to_msg()
            empty_path.header.frame_id = 'fsds/map'
            node.trajectory_pub.publish(empty_path)
            node.get_logger().info("Cleaning up and clearing path in RViz.")
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
