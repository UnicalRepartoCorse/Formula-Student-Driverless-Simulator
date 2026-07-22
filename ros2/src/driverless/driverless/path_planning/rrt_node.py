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
        self.declare_parameter('sample_radius_centerline', 1.35)
        self.declare_parameter('max_iter', 600)
        self.declare_parameter('num_trees', 1)
        self.declare_parameter('last_point', 4)
        self.declare_parameter('centerline_topic', '/track/centerline')
        self.declare_parameter('cones_topic', '/fsds/testing_only/track')

        # Read parameters
        collision_strategy = self.get_parameter('collision_strategy').value
        centerline_topic = self.get_parameter('centerline_topic').value
        cones_topic = self.get_parameter('cones_topic').value

        # State
        self.centerline = []  # list of (x, y) tuples in global frame
        self.car_x = 0.0
        self.car_y = 0.0
        self.car_yaw = 0.0
        # Cached trig for car_yaw — updated in odom_callback
        self._cos_yaw = 1.0
        self._sin_yaw = 0.0

        # --- Cone-triggered replanning state ---
        # Coni visti dall'inizio, usati per rilevare nuovi coni
        self.seen_blue_keys:   set = set()
        self.seen_yellow_keys: set = set()
        # Numero di coni visti all'ultimo plan
        self.n_blue_at_last_plan:   int = 0
        self.n_yellow_at_last_plan: int = 0
        # Soglia: quanti NUOVI coni per lato triggherano un replan
        self.NEW_CONE_THRESHOLD: int = 1
        # Ultimo punto del path precedente in frame GLOBALE (gx, gy, gtheta)
        self.last_goal_global = None
        # Ultima goal line calcolata in coordinate GLOBALI ((gx1, gy1), (gx2, gy2))
        self.last_goal_line_global = None
        # Path completo pubblicato in frame globale (lista di (gx, gy))
        self.published_path_global = []
        # Lista di stati globali (gx, gy, gtheta) del path precedente
        self.last_path_global_states = []

        # --- Multi-lap detection (start-cone re-encounter) ---
        # Salva i primi coni visti come riferimento di partenza.
        # Quando il veicolo si allontana (nessun cono di partenza visibile)
        # e poi li ri-incontra, significa che ha completato un giro.
        self._start_cone_keys: set = set()     # chiavi dei coni visti al primo frame
        self._start_cones_captured: bool = False  # True dopo il primo salvataggio
        self._left_start_zone: bool = False       # True quando nessun cono di partenza è più visibile
        self._MIN_START_REENCOUNTER: int = 2      # quanti coni di partenza ri-vedere per confermare il giro

        # Boundary caches
        self.last_blue_boundary = []
        self.last_yellow_boundary = []

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

    # ------------------------------------------------------------------
    # Coordinate transforms (cached trig for car_yaw)
    # ------------------------------------------------------------------

    def to_global(self, lx, ly):
        """Convert local (vehicle frame) coordinates to global frame."""
        gx = self.car_x + lx * self._cos_yaw - ly * self._sin_yaw
        gy = self.car_y + lx * self._sin_yaw + ly * self._cos_yaw
        return gx, gy

    def to_local(self, gx, gy):
        """Convert global coordinates to local (vehicle frame)."""
        dx = gx - self.car_x
        dy = gy - self.car_y
        lx =  dx * self._cos_yaw + dy * self._sin_yaw
        ly = -dx * self._sin_yaw + dy * self._cos_yaw
        return lx, ly

    def to_local_pose(self, gx, gy, gtheta):
        """Convert a global pose (x, y, theta) to local (vehicle frame)."""
        lx, ly = self.to_local(gx, gy)
        ltheta = normalize_angle(gtheta - self.car_yaw)
        return lx, ly, ltheta

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def centerline_callback(self, msg: Path):
        """
        Callback triggered when a new centerline is received.
        Stores centerline as list of (x, y) tuples — no Node objects needed.
        """
        self.centerline = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]

    def odom_callback(self, msg: Odometry):
        self.car_x = msg.pose.pose.position.x
        self.car_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.car_yaw = math.atan2(siny_cosp, cosy_cosp)
        # Cache trig — used by to_global, to_local, to_local_pose, _trim_published_path
        self._cos_yaw = math.cos(self.car_yaw)
        self._sin_yaw = math.sin(self.car_yaw)

    def cones_callback(self, msg: Track):
        self.last_track_msg = msg

    # ------------------------------------------------------------------
    # Main planning loop
    # ------------------------------------------------------------------

    def plan_timer_callback(self): #TODO rivedere con coni da perception
        if self.last_track_msg is None:
            return

        msg = self.last_track_msg
        cos_y = self._cos_yaw
        sin_y = self._sin_yaw

        # 1. Extract cone coordinates from msg and transform to local frame
        global_blue_cones = []
        global_yellow_cones = []
        global_orange_cones = []

        local_blue_cones = []
        local_yellow_cones = []
        local_orange_cones = []

        for c in msg.track:
            gx, gy = c.location.x, c.location.y
            dx = gx - self.car_x
            dy = gy - self.car_y
            # Squared distance check (avoid sqrt)
            if dx*dx + dy*dy < 100.0:  # 10.0²
                lx =  dx * cos_y + dy * sin_y
                ly = -dx * sin_y + dy * cos_y
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

        # 2. Chiave arrotondata per robustezza float e set dei coni attualmente visibili
        def cone_key(gx, gy): return (round(gx, 1), round(gy, 1))

        current_visible_keys = set()
        for gx, gy in global_blue_cones + global_yellow_cones:
            current_visible_keys.add(cone_key(gx, gy))

        # ---- LOGICA MULTI-GIRO (ri-incontro coni di partenza) ----
        # Fase 1: al primo frame con coni, salva i coni visibili come riferimento
        if not self._start_cones_captured and current_visible_keys:
            self._start_cone_keys = current_visible_keys.copy()
            self._start_cones_captured = True
            self.get_logger().info(
                f"Salvati {len(self._start_cone_keys)} coni di partenza come riferimento giro.")

        elif self._start_cones_captured and not self._left_start_zone:
            # Fase 2: controlla se il veicolo si è allontanato dalla zona di partenza
            # (nessun cono di partenza è più nel raggio di visibilità)
            overlap = current_visible_keys & self._start_cone_keys
            if len(overlap) == 0:
                self._left_start_zone = True
                self.get_logger().info(
                    "Zona di partenza lasciata — inizio monitoraggio per completamento giro.")

        elif self._left_start_zone:
            # Fase 3: controlla se il veicolo rivede i coni di partenza
            overlap = current_visible_keys & self._start_cone_keys
            if len(overlap) >= self._MIN_START_REENCOUNTER:
                self.get_logger().info(
                    f"Giro completato! Ri-incontrati {len(overlap)} coni di partenza. Reset memoria coni.")
                # Reset solo della memoria dei coni — il path e il goal restano
                # intatti per garantire continuità. Il replan avverrà naturalmente
                # perché i contatori azzerati triggerano should_replan = True.
                self.seen_blue_keys.clear()
                self.seen_yellow_keys.clear()
                self.n_blue_at_last_plan = 0
                self.n_yellow_at_last_plan = 0
                # Reset multi-lap: pronto per il prossimo giro
                self._left_start_zone = False

        # Aggiorna i coni visti cumulativamente
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

        # 3. Centerline in frame locale (list of (x, y) tuples)
        local_centerline = []
        if self.centerline:
            for cx, cy in self.centerline:
                lx, ly = self.to_local(cx, cy)
                local_centerline.append((lx, ly))

        # 4. Goal line in frame locale
        goal_line = None

        if local_centerline and len(local_centerline) > 1:
            p1x, p1y = local_centerline[-2]
            p2x, p2y = local_centerline[-1]
            dx = p2x - p1x
            dy = p2y - p1y
            length = math.hypot(dx, dy)
            if length > 0:
                nx = -dy / length * 2.0
                ny =  dx / length * 2.0
                goal_line = ((p2x + nx, p2y + ny), (p2x - nx, p2y - ny))

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

        # 5. Start del nuovo RRT: prendiamo il N-esimo punto dalla fine del path precedente
        #    (se disponibile) per garantire una transizione liscia,
        #    altrimenti partiamo dall'origine (posizione auto).
        last_point = self.get_parameter('last_point').value
        rrt_start, overlap_count = self._compute_rrt_start(last_point)

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

            # Rimuovi gli ultimi N elementi sovrapposti per collegare correttamente il nuovo percorso
            if 0 < overlap_count <= len(self.published_path_global):
                self.published_path_global = self.published_path_global[:-overlap_count]

            self.published_path_global.extend(new_path_global)

            # Salva l'ultimo punto in globale per fallback
            last_local = new_path_local[-1]
            lgx, lgy   = self.to_global(last_local[0], last_local[1])
            lgtheta    = normalize_angle(last_local[2] + self.car_yaw)
            self.last_goal_global = (lgx, lgy, lgtheta)

            # Salva tutti i nuovi stati globali per il prossimo replan
            self.last_path_global_states = []
            for p in new_path_local:
                gx, gy = self.to_global(p[0], p[1])
                gtheta = normalize_angle(p[2] + self.car_yaw)
                self.last_path_global_states.append((gx, gy, gtheta))

            # Aggiorna soglia coni
            self.n_blue_at_last_plan   = len(self.seen_blue_keys)
            self.n_yellow_at_last_plan = len(self.seen_yellow_keys)

        # 8. Pubblica e visualizza
        self.publish_path(self.published_path_global)
        self.publish_viz(self.published_path_global, rrt.node_list, self.last_goal_line_global,
                         global_blue_cones, global_yellow_cones, global_orange_cones, rrt.sampled_points)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_rrt_start(self, last_point):
        """
        Compute the RRT start pose in local frame from the previous path.
        Returns (rrt_start_tuple, overlap_count).
        """
        if self.last_path_global_states and len(self.last_path_global_states) >= last_point:
            gx, gy, gtheta = self.last_path_global_states[-last_point]
            lx, ly, ltheta = self.to_local_pose(gx, gy, gtheta)
            return (lx, ly, ltheta), last_point
        elif self.last_goal_global is not None:
            gx, gy, gtheta = self.last_goal_global
            lx, ly, ltheta = self.to_local_pose(gx, gy, gtheta)
            return (lx, ly, ltheta), 0
        else:
            return (0.0, 0.0, 0.0), 0

    def _trim_published_path(self):
        """Rimuove i waypoint già percorsi (più di 3 m dietro la macchina)."""
        if not self.published_path_global:
            return
        cos_y = self._cos_yaw
        sin_y = self._sin_yaw
        car_x = self.car_x
        car_y = self.car_y
        trimmed = []
        for gx, gy in self.published_path_global:
            dx = gx - car_x
            dy = gy - car_y
            lx = dx * cos_y + dy * sin_y  # coordinata longitudinale locale
            if lx > -3.0:  # tieni tutto ciò che non è più di 3 m dietro
                trimmed.append((gx, gy))
        self.published_path_global = trimmed

    # ------------------------------------------------------------------
    # Visualization helpers
    # ------------------------------------------------------------------

    def _make_marker(self, ns, marker_id, marker_type, color_rgba, scale, now, lifetime):
        """
        Factory for common Marker fields. Reduces boilerplate in publish_viz.
        color_rgba: (r, g, b, a)   scale: (x,) or (x, y) or (x, y, z)
        """
        m = Marker()
        m.header.frame_id = "fsds/map"
        m.header.stamp = now
        m.ns = ns
        m.id = marker_id
        m.type = marker_type
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = scale[0]
        m.scale.y = scale[1] if len(scale) > 1 else scale[0]
        m.scale.z = scale[2] if len(scale) > 2 else 0.0
        m.color.r = color_rgba[0]
        m.color.g = color_rgba[1]
        m.color.b = color_rgba[2]
        m.color.a = color_rgba[3]
        m.lifetime = lifetime
        return m

    def publish_viz(self, path, nodes, goal, blue_cones, yellow_cones, orange_cones, samples=None):
        """
        Publishes markers for the RRT* tree, the final path, the goal, and the centerline.
        """
        marker_array = MarkerArray()
        now = self.get_clock().now().to_msg()
        lifetime_msg = Duration(sec=400, nanosec=50) # Aumentato a 1.5s per eliminare lo sfarfallio

        # 1. Tree Marker (Line List)
        tree_marker = self._make_marker("rrt_tree", 0, Marker.LINE_LIST,
                                        (0.6, 0.1, 0.8, 0.6), (0.02,), now, lifetime_msg)
        for node in nodes:
            if node.parent is not None:
                gx1, gy1 = self.to_global(node.parent.x, node.parent.y)
                gx2, gy2 = self.to_global(node.x, node.y)
                tree_marker.points.append(Point(x=float(gx1), y=float(gy1), z=0.0))
                tree_marker.points.append(Point(x=float(gx2), y=float(gy2), z=0.0))
        marker_array.markers.append(tree_marker)

        # 1b. Tree Nodes Marker (Points, light blue)
        if nodes:
            tree_nodes_marker = self._make_marker("rrt_tree_nodes", 15, Marker.POINTS,
                                                  (0.6, 0.1, 0.8, 0.2), (0.12, 0.12), now, lifetime_msg)
            for node in nodes:
                gx, gy = self.to_global(node.x, node.y)
                tree_nodes_marker.points.append(Point(x=float(gx), y=float(gy), z=0.0))
            marker_array.markers.append(tree_nodes_marker)

        # 1c. Sampled Points Marker (Points, semi-transparent orange)
        if samples:
            samples_marker = self._make_marker("rrt_samples", 20, Marker.POINTS,
                                               (1.0, 0.4, 0.0, 0.2), (0.15, 0.08), now, lifetime_msg)
            for sx, sy in samples:
                gx, gy = self.to_global(sx, sy)
                samples_marker.points.append(Point(x=float(gx), y=float(gy), z=0.0))
            marker_array.markers.append(samples_marker)

        # 2. Path Marker (Line Strip)
        if path:
            path_marker = self._make_marker("rrt_path", 1, Marker.LINE_STRIP,
                                            (0.0, 1.0, 0.0, 1.0), (0.1,), now, lifetime_msg)
            for p in path:
                path_marker.points.append(Point(x=float(p[0]), y=float(p[1]), z=0.0))
            marker_array.markers.append(path_marker)

            # 2b. Path Points Marker (Sphere List)
            path_points_marker = self._make_marker("rrt_path_points", 10, Marker.SPHERE_LIST,
                                                   (1.0, 0.0, 0.0, 1.0), (0.15, 0.15, 0.15), now, lifetime_msg)
            for p in path:
                path_points_marker.points.append(Point(x=float(p[0]), y=float(p[1]), z=0.0))
            marker_array.markers.append(path_points_marker)

        # 3. Goal Marker (solo durante replan, quando goal non è None)
        if goal is not None:
            goal_marker = self._make_marker("rrt_goal", 2, Marker.LINE_STRIP,
                                            (1.0, 0.0, 1.0, 1.0), (0.1,), now, lifetime_msg)
            goal_marker.points.append(Point(x=float(goal[0][0]), y=float(goal[0][1]), z=0.0))
            goal_marker.points.append(Point(x=float(goal[1][0]), y=float(goal[1][1]), z=0.0))
        else:
            goal_marker = Marker()
            goal_marker.header.frame_id = "fsds/map"
            goal_marker.header.stamp = now
            goal_marker.ns = "rrt_goal"
            goal_marker.id = 2
            goal_marker.action = Marker.DELETE
        marker_array.markers.append(goal_marker)

        # 4. Centerline Marker (Line Strip, Red)
        if self.centerline:
            cl_marker = self._make_marker("centerline_ref", 3, Marker.LINE_STRIP,
                                          (1.0, 0.0, 0.0, 0.8), (0.05,), now, lifetime_msg)
            for cx, cy in self.centerline:
                cl_marker.points.append(Point(x=float(cx), y=float(cy), z=0.0))
            marker_array.markers.append(cl_marker)

        # 4b. Blue Boundary Marker (Line Strip, Blue)
        if self.last_blue_boundary:
            blue_b_marker = self._make_marker("blue_boundary", 50, Marker.LINE_STRIP,
                                              (0.0, 0.4, 1.0, 0.8), (0.05,), now, lifetime_msg)
            for p in self.last_blue_boundary:
                blue_b_marker.points.append(Point(x=float(p[0]), y=float(p[1]), z=0.0))
            marker_array.markers.append(blue_b_marker)

        # 4c. Yellow Boundary Marker (Line Strip, Yellow)
        if self.last_yellow_boundary:
            yellow_b_marker = self._make_marker("yellow_boundary", 51, Marker.LINE_STRIP,
                                                (1.0, 0.8, 0.0, 0.8), (0.05,), now, lifetime_msg)
            for p in self.last_yellow_boundary:
                yellow_b_marker.points.append(Point(x=float(p[0]), y=float(p[1]), z=0.0))
            marker_array.markers.append(yellow_b_marker)

        # 5. Cones Markers (CUBE_LIST for efficiency)
        def add_cones_marker(cones, marker_id, r, g, b):
            if not cones:
                return
            m = self._make_marker("rrt_cones", marker_id, Marker.CUBE_LIST,
                                  (r, g, b, 1.0), (0.2, 0.2, 0.3), now, lifetime_msg)
            for cx, cy in cones:
                m.points.append(Point(x=float(cx), y=float(cy), z=0.15))
            marker_array.markers.append(m)

        add_cones_marker(blue_cones, 4, 0.0, 0.0, 1.0) # Blue
        add_cones_marker(yellow_cones, 5, 1.0, 1.0, 0.0) # Yellow
        add_cones_marker(orange_cones, 6, 1.0, 0.5, 0.0) # Orange

        self.viz_pub.publish(marker_array)

    # ------------------------------------------------------------------
    # Publishers
    # ------------------------------------------------------------------

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

        def _build_boundary_msg(boundary):
            boundary_msg = Path()
            boundary_msg.header.stamp = now
            boundary_msg.header.frame_id = 'fsds/map'
            if boundary:
                for p in boundary:
                    pose = PoseStamped()
                    pose.header = boundary_msg.header
                    pose.pose.position.x = float(p[0])
                    pose.pose.position.y = float(p[1])
                    pose.pose.position.z = 0.0
                    pose.pose.orientation.w = 1.0
                    boundary_msg.poses.append(pose)
            return boundary_msg

        self.blue_boundary_pub.publish(_build_boundary_msg(self.last_blue_boundary))
        self.yellow_boundary_pub.publish(_build_boundary_msg(self.last_yellow_boundary))

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
