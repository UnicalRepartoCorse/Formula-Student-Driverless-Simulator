import math
import random
from math import inf
from typing import List, Tuple, Optional
from driverless.utils.collision_checker import CollisionChecker
from driverless.utils.utils import normalize_angle


class Node:
    """
    Represents a node in the RRT* tree.
    """
    def __init__(self, x: float, y: float, theta: float):
        self.x = x
        self.y = y
        self.theta = theta
        self.cost = 0.0
        self.parent = None
        #storing dist to goal for efficiency
        self.dist_to_goal = inf
        # Store the path of (x, y) coordinates from the parent to this node
        self.path = []

class KinematicRRTStar:
    """
    Kinematic RRT* algorithm for path planning.
    Takes into account the maximum steering angle of a vehicle.
    """

    def __init__(self, start: Tuple[float, float, float], goal_line: Tuple[Tuple[float, float], Tuple[float, float]],
                 bounds: Tuple[float, float, float, float], collision_checker: CollisionChecker,
                 max_iter: int = 500, step_size: float = 1.0,
                 max_steering_angle: float = 0.4, wheelbase: float = 1.5,
                 sample_radius_centerline: float = 1.4,
                 centerline: Optional[List[Tuple[float, float]]] = None,
                 rrt_targets: Optional[List[Tuple[float, float, float]]] = None):
        """
        Initialize the RRT* planner.
        
        Args:
            start: Starting pose (x, y, theta).
            goal_line: Goal line segment ((x1, y1), (x2, y2)).
            bounds: Sampling boundaries (x_min, x_max, y_min, y_max).
            collision_checker: Instance of CollisionChecker.
            max_iter: Maximum number of iterations.
            step_size: Distance to extend the tree at each step.
            max_steering_angle: Maximum steering angle of the vehicle in radians.
            wheelbase: Wheelbase of the vehicle in meters.
            centerline: Optional list of (x, y) coordinates representing the track centerline.
        """
        self.start = Node(start[0], start[1], start[2])
        self.goal_line = goal_line
        self.bounds = bounds
        self.collision_checker = collision_checker

        self.max_iter = max_iter
        self.step_size = step_size
        self.max_steering_angle = max_steering_angle
        self.wheelbase = wheelbase
        self.centerline = centerline
        self.rrt_targets = rrt_targets

        self.node_list = [self.start]
        self.sampled_points = []  # Memorizza tutti i punti campionati ad ogni passo
        self.sample_radius_centerline = sample_radius_centerline #m

    def _dist_to_goal_line(self, x, y):
        """
        Calculates the shortest distance from a point (x, y) to the goal line segment.
        If the point is past the goal line, it returns 0.0.
        """
        a, b = self.goal_line
        ax, ay = a
        bx, by = b
        dx = bx - ax
        dy = by - ay

        # 1. Calcola il punto più vicino sul segmento goal_line
        if dx == 0 and dy == 0:
            dist_seg = math.hypot(x - ax, y - ay)
        else:
            t = ((x - ax) * dx + (y - ay) * dy) / (dx * dx + dy * dy)
            t = max(0.0, min(1.0, t))
            proj_x = ax + t * dx
            proj_y = ay + t * dy
            dist_seg = math.hypot(x - proj_x, y - proj_y)

        # 2. Verifica se il punto ha superato la goal line
        # Punto medio del traguardo
        mx = (ax + bx) / 2.0
        my = (ay + by) / 2.0

        # Vettore dal traguardo al punto
        wx = x - mx
        wy = y - my

        # Vettore orientamento goal line (da A a B)
        ux = bx - ax
        uy = by - ay

        # Vettore normale provvisorio alla goal line
        nx = -uy
        ny = ux

        # Vettore direzione di marcia iniziale (dallo start al midpoint)
        fx = mx - self.start.x
        fy = my - self.start.y

        # Allinea la normale nella direzione di marcia
        if (nx * fx + ny * fy) < 0:
            nx = -nx
            ny = -ny

        # Prodotto scalare per determinare se il punto è oltre la linea
        is_past = (wx * nx + wy * ny) > 0

        if is_past:
            return 0.0

        return dist_seg

    def plan(self) -> Optional[List[Tuple[float, float, float]]]:
        """
        Execute the RRT* planning algorithm.
        
        Returns:
            List of (x, y, theta) representing the path, or None if no path found.
        """
        for i in range(0, self.max_iter):

            sample = self._sample_free_space() # Node: x, y, theta
            self.sampled_points.append((sample.x, sample.y)) #ONLY FOR GRAPHICS

            nearest_index = self._get_nearest_node_index(self.node_list, sample)
            nearest_node = self.node_list[nearest_index]

            new_node = self._steer(nearest_node, sample)

            # no reachable
            if new_node is None:
                continue

            new_node.parent = nearest_node
            new_node.cost = self._calc_new_cost(nearest_node, new_node)
            new_node.dist_to_goal = self._dist_to_goal_line(new_node.x, new_node.y)

            # collision check
            if not self.collision_checker.is_path_free(new_node.path):
                continue

            nearest_indeces = self._get_near_nodes(new_node)
            new_node = self._choose_parent(new_node, nearest_indeces) #update parent, cost and path of the new node

            self.node_list.append(new_node)

            #self._rewire(new_node, nearest_indeces)


        # Selezione del percorso migliore
        # Filtra tutti i nodi che hanno raggiunto o superato il traguardo
        goal_nodes = [node for node in self.node_list if node.dist_to_goal <= 0]

        if goal_nodes:
            # Tra quelli al traguardo, scegliamo quello con il costo minore
            best_node = min(goal_nodes, key=lambda n: n.cost)
        else:
            # Fallback: se nessuno ha raggiunto il traguardo, prendiamo il più vicino
            best_node = min(self.node_list, key=lambda n: n.dist_to_goal)

        if best_node is not None:
            return self._extract_path(best_node)
        else:
            return None

    #CORRECT
    def _sample_free_space(self) -> Node:
        """
        Randomly sample a point (x, y, theta).
        If a centerline is provided, strictly sample within a 1.5m radius 
        of a randomly chosen point on that centerline.
        Otherwise, sample uniformly within the bounds.
        """
        # Target-biased sampling (comportamento get_random_point_from_target_list)
        if self.rrt_targets and len(self.rrt_targets) > 0:
            target_id = random.randint(0, len(self.rrt_targets) - 1)
            tx, ty, o_size = self.rrt_targets[target_id]

            rand_angle = random.uniform(0, 2 * math.pi)
            rand_dist = random.uniform(o_size, 3.0)  # maxTargetAroundDist = 3
            x = tx + rand_dist * math.cos(rand_angle)
            y = ty + rand_dist * math.sin(rand_angle)
            theta = random.uniform(-math.pi, math.pi)
            return Node(x, y, theta)

        if self.centerline and len(self.centerline) > 0:
            # Pick a random reference segment on the centerline
            i = random.randint(0, len(self.centerline) - 2)
            ref_pt = self.centerline[i]
            ref_pt2 = self.centerline[i + 1]

            # Sample within a circle of radius R around the reference point
            r = self.sample_radius_centerline * math.sqrt(random.uniform(0, 1))  # sqrt for uniform distribution
            alpha = random.uniform(0, 2 * math.pi)
            x = ref_pt.x + r * math.cos(alpha)
            y = ref_pt.y + r * math.sin(alpha)

            # Local centerline heading: θᵢ = atan2(yᵢ₊₁ - yᵢ, xᵢ₊₁ - xᵢ)
            theta = math.atan2(ref_pt2.y - ref_pt.y, ref_pt2.x - ref_pt.x)
            theta = theta + random.uniform(-self.max_steering_angle, self.max_steering_angle)
            theta = normalize_angle(theta)
            return Node(x, y, theta)

        else:
            x_min, x_max, y_min, y_max = self.bounds
            x = random.uniform(x_min, x_max)
            y = random.uniform(y_min, y_max)
            theta = random.uniform(-math.pi, math.pi)
            return Node(x, y, theta)


    def _steer(self, from_node: Node, target: Node) -> Optional[Node]:
        """
        Generate a kinematically feasible trajectory from from_node toward to_point
        using a bicycle model and Pure Pursuit inspired steering.

        Returns:
            A new Node if a valid motion is generated, otherwise None.
        """
        ds = 0.2                       # passo di integrazione [m]
        goal_tolerance = self.step_size             # distanza per considerare raggiunto il target

        x = from_node.x
        y = from_node.y
        theta = from_node.theta

        cost = from_node.cost
        travelled = 0.0
        path = []

        while travelled < ds:

            # Vettore verso il target
            dx = target.x - x
            dy = target.y - y

            distance = math.hypot(dx, dy)

            # Se siamo arrivati abbastanza vicini
            if distance < goal_tolerance:
                break

            # Direzione verso il target
            heading = math.atan2(dy, dx)

            # Errore angolare [-pi, pi]
            alpha = heading - theta
            alpha = math.atan2(math.sin(alpha), math.cos(alpha))

            # Lookahead
            Ld = max(distance, 1.0)

            # Curvatura Pure Pursuit
            kappa = 2.0 * math.sin(alpha) / Ld

            # Steering richiesto
            delta = math.atan(self.wheelbase * kappa)

            # Saturazione
            delta = max(
                -self.max_steering_angle,
                min(delta, self.max_steering_angle)
            )

            # Integrazione modello bicycle
            theta += ds * math.tan(delta) / self.wheelbase
            theta = normalize_angle(theta)

            x += ds * math.cos(theta)
            y += ds * math.sin(theta)

            travelled += ds

            # Costo del segmento
            cost += ds * (1.0 + 0.2 * (kappa ** 2))

            path.append((x, y, theta))

        # Se non siamo riusciti a muoverci
        if travelled < 1e-3:
            return None

        new_node = Node(x, y, theta)
        new_node.parent = from_node
        new_node.path = path
        new_node.cost = cost

        return new_node



    def _propagate_cost_to_children(
            self,
            parent_node: Node,
            old_cost: float,
            new_cost: float
    ):
        """
        Propagate cost updates to descendants after rewiring using an iterative BFS approach.
        This prevents RecursionError and handles potential cycles gracefully.
        """
        delta = new_cost - old_cost

        # Iterative BFS using a queue
        queue = [parent_node]
        visited = {parent_node}

        while queue:
            current_parent = queue.pop(0)

            for node in self.node_list:
                if node.parent == current_parent:
                    if node in visited:
                        # Cycle detected! Break the cycle to prevent infinite loops
                        node.parent = None
                        continue

                    node.cost += delta
                    visited.add(node)
                    queue.append(node)

    def _calc_new_cost(self, from_node: Node, to_node: Node) -> float:
        """
        Calculate the cost of moving from from_node to to_node.
        Includes a curvature penalty to encourage straighter paths,
        """
        path_cost = 0.0
        path = to_node.path
        if len(path) > 0:

            # Aggiunge la distanza tra il nodo parent e il primo punto della traiettoria simulata
            path_cost += math.hypot(from_node.x - path[0][0], from_node.y - path[0][1])
            # Aggiunge la distanza tra tutti i punti consecutivi
            for i in range(len(path)-1): # depend on the number of iter in the steer path
                p1 = path[i]
                p2 = path[i+1]
                path_cost += math.hypot(p1[0] - p2[0], p1[1] - p2[1])
        else:
            path_cost += math.hypot(from_node.x - to_node.x, from_node.y - to_node.y)


        # Centerline Penalty: Heavily penalize deviating from the track center
        cl_penalty = 0.0
        if self.centerline and len(self.centerline) > 0:
            ind_near_center_pt = self._get_nearest_node_index(self.centerline, from_node)
            if ind_near_center_pt == len(self.centerline) - 1:
                ind_near_center_pt -= 1
            
            p_cl1 = self.centerline[ind_near_center_pt]
            p_cl2 = self.centerline[ind_near_center_pt + 1]
            
            # Centerline vector (from point i to i+1)
            v_cl_x = p_cl2.x - p_cl1.x
            v_cl_y = p_cl2.y - p_cl1.y
            
            # Node vector (from from_node to to_node)
            v_node_x = to_node.x - from_node.x
            v_node_y = to_node.y - from_node.y
            
            norm_cl = math.hypot(v_cl_x, v_cl_y)
            norm_node = math.hypot(v_node_x, v_node_y)
            
            if norm_cl > 1e-6 and norm_node > 1e-6:
                cos_sim = (v_cl_x * v_node_x + v_cl_y * v_node_y) / (norm_cl * norm_node)
            else:
                cos_sim = 1.0


            cl_penalty = 1.0 - abs(cos_sim)

        total = from_node.cost + path_cost +  10 * cl_penalty #TODO VALUATE THE PENALTY COEFFICIENT
        return total

    def _get_near_nodes(self, new_node: Node) -> List[int]:
        """
        Find all nodes in the tree within a certain radius of new_node (for rewiring).
        """

        radius = self.step_size * 5 # TODO da modificare con andamento logaritmico
        near_indices= []

        for i in range(len(self.node_list)):
            x_n, y_n = new_node.x, new_node.y
            x, y = self.node_list[i].x, self.node_list[i].y

            dist = math.hypot(x - x_n, y - y_n)
            if dist < radius:
                near_indices.append(i)

        return near_indices

    def _get_nearest_node_index(self, node_list: List[Node], rnd_node: Node) -> int:
        """
        Find the index of the nearest node in the tree to the sampled point.
        """
        best= +inf
        best_index=-1
        for i in range(len(node_list)):
            node= node_list[i]
            dist = math.hypot(node.x-rnd_node.x, node.y-rnd_node.y)
            if dist < best:
                best_index = i
                best=dist
        return best_index



    def _extract_path(self, last_node: Node):
        # Risali la catena di nodi
        nodes = []
        curr = last_node
        while curr is not None:
            nodes.append(curr)
            curr = curr.parent
        nodes.reverse()

        # Concatena i path cinematici intermedi di ogni nodo
        dense = [(nodes[0].x, nodes[0].y, nodes[0].theta)]
        for node in nodes[1:]:
            dense.extend(node.path)  # (x, y, theta) da _steer

        # --- OPZIONI PER IL PATH FINALE ---
        # Opzione 1 (Attiva): Percorso grezzo puro (nessuno smoothing)
        return dense, last_node.cost

        # Opzione 2 (Disattivata): Percorso levigato con spline cubica (campionamento equispaziato ad arco)
        #return self._smooth_path(dense, last_node.cost)

    def _smooth_path(
            self, path: List[Tuple[float, float, float]], path_cost: float
    ) -> Tuple[List[Tuple[float, float, float]], float]:
        import numpy as np
        from scipy.interpolate import splprep, splev

        if len(path) < 4:
            return path, path_cost

        pts = np.array([(p[0], p[1]) for p in path])

        try:
            # Fit parametrico della spline
            tck, u = splprep([pts[:, 0], pts[:, 1]], s=len(path) * 0.5, k=3)
        except Exception:
            return path, path_cost

        # 1. Campioniamo la spline in modo molto denso (1000 punti) per mappare lo spazio
        u_dense = np.linspace(0.0, 1.0, 1000)
        x_dense, y_dense = splev(u_dense, tck)

        # 2. Calcoliamo la lunghezza d'arco cumulativa (in metri) lungo i punti densi
        dx = np.diff(x_dense)
        dy = np.diff(y_dense)
        ds = np.hypot(dx, dy)
        s = np.concatenate([[0.0], np.cumsum(ds)])
        total_len = s[-1]

        if total_len < 1e-6:
            return path, path_cost

        # 3. Definiamo la risoluzione desiderata (distanza esatta costante tra i punti, es. 0.2 m)
        n_out = max(10, int(total_len / self.step_size))

        # 4. Creiamo punti di campionamento perfettamente equispaziati in metri (lunghezza d'arco)
        s_targets = np.linspace(0.0, total_len, n_out)

        # 5. Interpoliamo le coordinate X e Y basandoci sulla coordinata curvilinea s
        x_s = np.interp(s_targets, s, x_dense)
        y_s = np.interp(s_targets, s, y_dense)

        # 6. Calcoliamo l'angolo theta (heading) in modo coerente
        dx_dt = np.gradient(x_s)
        dy_dt = np.gradient(y_s)
        theta_s = np.arctan2(dy_dt, dx_dt)

        smoothed = [(float(x_s[i]), float(y_s[i]), float(theta_s[i])) for i in range(n_out)]
        if len(smoothed) > 0:
            # Preserva l'ancoraggio perfetto del punto iniziale
            smoothed[0] = path[0]
        return smoothed, path_cost