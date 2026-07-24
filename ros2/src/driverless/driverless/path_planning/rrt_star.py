import math
import random
from collections import deque
from math import inf, atan2
from typing import List, Tuple, Optional

import numpy as np
from scipy.interpolate import splprep, splev
from scipy.spatial import cKDTree

from driverless.utils.collision_checker import CollisionChecker
from driverless.utils.utils import normalize_angle

'''
The state of a the car in a 2-D Cartesian coordinates is defined as S = (x, y, θ), and the kinematic model is described by
the following equation
    - x = v · cos φ · cos θ
    - y = v · cos φ · sin θ
    - θ = v · sin φ/l
where v is the driving velocity of the rear wheels, θ is the orientation angle of the robot body with respect to the X-axis, φ is the steering angle of the front wheels, and l is the wheelbase
Turning radius ρ ≥ ρmin = l/ tan φmax
'''

class Node:
    """
    Represents a node in the RRT* tree.
    """
    __slots__ = ('x', 'y', 'theta', 'cos_theta', 'sin_theta',
                 'cost', 'parent', 'children', 'dist_to_goal', 'path')

    def __init__(self, x: float, y: float, theta: float):
        self.x = x
        self.y = y
        self.theta = theta
        # Cache trigonometric functions of theta to avoid repeating math.cos/math.sin calls
        self.cos_theta = math.cos(theta)
        self.sin_theta = math.sin(theta)
        self.cost = 0.0
        self.parent = None
        self.children = []   # childer list for efficiency
        self.dist_to_goal = inf # storing dist to goal for efficiency
        self.path = [] # Store the path of (x, y, theta) coordinates from the parent to this node

class RRTStar:
    """
    Kinematic RRT* algorithm for path planning.
    Takes into account the maximum steering angle of a vehicle.
    """

    def __init__(self,
                 start: Tuple[float, float, float],
                 goal_line: Tuple[Tuple[float, float], Tuple[float, float]],
                 collision_checker: CollisionChecker,
                 max_iter: int = 500,
                 step_size: float = 1.0,
                 max_steering_angle: float = 0.4,
                 wheelbase: float = 1.5,
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
        self.collision_checker = collision_checker

        self.max_iter = max_iter
        self.step_size = step_size
        self.max_steering_angle = max_steering_angle
        self.wheelbase = wheelbase
        self._inv_wheelbase = 1.0 / wheelbase
        if centerline is not None:
            self.centerline = np.array([[pt.x, pt.y] for pt in centerline])
            self._cl_tree = cKDTree(self.centerline)
        else:
            self.centerline = []
            self._cl_tree = None
        self.rrt_targets = rrt_targets

        self.node_list = [self.start]
        self.sampled_points = []  # JUST FOR GRAPHIC DESIGN
        self.sample_radius_centerline = sample_radius_centerline #m

        # Preallocate numpy array for node coordinates to enable vectorized distance checks
        self.node_coords = np.zeros((self.max_iter + 1, 2))
        self.node_coords[0] = [self.start.x, self.start.y]



            # Goal line precomputations
        a, b = self.goal_line
        self.ax, self.ay = a
        self.bx, self.by = b
        self.goal_dx = self.bx - self.ax
        self.goal_dy = self.by - self.ay
        self.goal_len_sq = self.goal_dx * self.goal_dx + self.goal_dy * self.goal_dy

        self.goal_mx = (self.ax + self.bx) / 2.0
        self.goal_my = (self.ay + self.by) / 2.0
        self.goal_nx = -self.goal_dy
        self.goal_ny = self.goal_dx
        fx = self.goal_mx - self.start.x
        fy = self.goal_my - self.start.y
        if (self.goal_nx * fx + self.goal_ny * fy) < 0:
            self.goal_nx = -self.goal_nx
            self.goal_ny = -self.goal_ny


    def _dist_to_goal_line(self, x, y):
        """
        Calculates the shortest distance from a point (x, y) to the goal line segment.
        If the point is past the goal line, it returns 0.0.
        """
        # 1. Calcola il punto più vicino sul segmento goal_line
        if self.goal_len_sq == 0:
            dist_seg = math.hypot(x - self.ax, y - self.ay)
        else:
            t = ((x - self.ax) * self.goal_dx + (y - self.ay) * self.goal_dy) / self.goal_len_sq
            t = max(0.0, min(1.0, t))
            proj_x = self.ax + t * self.goal_dx
            proj_y = self.ay + t * self.goal_dy
            dist_seg = math.hypot(x - proj_x, y - proj_y)

        # 2. Verifica se il punto ha superato la goal line
        wx = x - self.goal_mx
        wy = y - self.goal_my

        # Prodotto scalare per determinare se il punto è oltre la linea
        is_past = (wx * self.goal_nx + wy * self.goal_ny) > 0

        if is_past:
            return 0.0

        return dist_seg

    def plan(self) -> Optional[Tuple[List[Tuple[float, float, float]], float]]:
        """
        Execute the RRT* planning algorithm.

        Returns:
            Tuple of (path, cost) where path is a list of (x, y, theta), or None if no path found.
        """
        for i in range(0, self.max_iter):

            sample = self._sample_free_space() # Node: x, y, theta
            self.sampled_points.append((sample.x, sample.y)) #ONLY FOR GRAPHICS

            nearest_index = self._get_nearest_node_index(self.node_list, sample)
            nearest_node = self.node_list[nearest_index]

            new_node = self._steer(nearest_node, sample, self.step_size)

            #new_node.parent = nearest_node
            new_node.cost = self._calc_new_cost(nearest_node, new_node)
            new_node.dist_to_goal = self._dist_to_goal_line(new_node.x, new_node.y)

            # collision check (fixed list type mismatch)
            if not self.collision_checker.is_path_free(new_node.path):
                continue

            nearest_indeces = self._get_near_nodes(new_node)

            new_node = self._choose_parent(new_node, nearest_indeces) #update parent, cost and path of the new node
            self.node_list.append(new_node)
            # Maintain node_coords
            self.node_coords[len(self.node_list) - 1] = [new_node.x, new_node.y]

            # Register as child of parent for efficient cost propagation
            if new_node.parent is not None:
                new_node.parent.children.append(new_node)

            self._rewire(new_node, nearest_indeces)


        # Selezione del percorso migliore
        # Filtra tutti i nodi che hanno raggiunto o superato il traguardo
        goal_nodes = [node for node in self.node_list if node.dist_to_goal <= 0]

        if goal_nodes:
            # Tra quelli al traguardo, scegliamo quello con il costo minore
            best_node = min(goal_nodes, key=lambda n: n.cost)
        else:
            # Fallback: se nessuno ha raggiunto il traguardo, prendiamo il più vicino
            best_node = min(self.node_list, key=lambda n: n.dist_to_goal)

        return self._extract_path(best_node)

    def _sample_free_space(self) -> Node:
        """
        Randomly sample a point (x, y, theta).
        If a centerline is provided, strictly sample within a 1.5m radius
        of a randomly chosen point on that centerline.
        Otherwise, sample uniformly within the bounds.
        """
        # Target-biased sampling (comportamento get_random_point_from_target_list)
        #if self.rrt_targets and len(self.rrt_targets) > 0:
        #    target_id = random.randint(0, len(self.rrt_targets) - 1)
        #    tx, ty, o_size = self.rrt_targets[target_id]
        #    rand_angle = random.uniform(0, 2 * math.pi)
        #    rand_dist = random.uniform(o_size, 3.0)  # maxTargetAroundDist = 3
        #    x = tx + rand_dist * math.cos(rand_angle)
        #    y = ty + rand_dist * math.sin(rand_angle)
        #    theta = random.uniform(-math.pi, math.pi)
        #    return Node(x, y, theta)

        if len(self.centerline) > 0:
            # Pick a random reference segment on the centerline
            i = random.randint(0, len(self.centerline) - 2)
            ref_pt = self.centerline[i]

            # Sample within a circle of radius R around the reference point
            r = self.sample_radius_centerline * math.sqrt(random.uniform(0, 1))  # sqrt for uniform distribution
            alpha = random.uniform(0, 2 * math.pi)
            x = ref_pt[0] + r * math.cos(alpha)
            y = ref_pt[1] + r * math.sin(alpha)

            ref_pt2 = self.centerline[i + 1]
            theta = math.atan2(ref_pt2[1] - ref_pt[1], ref_pt2[0] - ref_pt[0])
            theta = normalize_angle(theta)
            return Node(x, y, theta)

        else:
            x_min, x_max, y_min, y_max = self.collision_checker.xy_limit()
            x = random.uniform(x_min, x_max)
            y = random.uniform(y_min, y_max)
            theta = random.uniform(-math.pi, math.pi)
            return Node(x, y, theta)

    def _steer(self, nearest_node: Node, sample: Node, max_step: float) -> Node:
        # Calcola la distanza tra nearest_node e sample
        dist = math.hypot(sample.x - nearest_node.x, sample.y - nearest_node.y)

        # Limita il passo a max_step
        step_len = min(dist, max_step)

        theta = math.atan2(sample.y - nearest_node.y, sample.x - nearest_node.x)
        angleChange = normalize_angle(theta - nearest_node.theta)

        # Usa l'angolo di sterzo continuo, limitato ai confini fisici max_steering_angle
        steering_angle = max(-self.max_steering_angle, min(self.max_steering_angle, angleChange))

        # Equazioni cinematiche modello bicicletta (asse posteriore) - using cached trig functions
        x_dot = nearest_node.cos_theta
        y_dot = nearest_node.sin_theta
        theta_dot = math.tan(steering_angle) * self._inv_wheelbase

        # Integrazione modello discreto — crea Node direttamente con valori finali
        new_x = nearest_node.x + x_dot * step_len
        new_y = nearest_node.y + y_dot * step_len
        new_theta = normalize_angle(nearest_node.theta + theta_dot * step_len)

        newNode = Node(new_x, new_y, new_theta)
        newNode.cost = nearest_node.cost + step_len
        newNode.parent = nearest_node
        newNode.path = [(new_x, new_y, new_theta)]

        return newNode

    def _propagate_cost_to_children(self, parent_node: Node, old_cost: float, new_cost: float):
        """
        Propagate cost delta to all descendants using children lists.
        O(subtree_size) instead of O(n * subtree_depth).
        Handles potential cycles by tracking visited nodes.
        """
        delta = new_cost - old_cost

        queue = deque(parent_node.children)
        visited = {parent_node}

        while queue:
            node = queue.popleft()
            if node in visited:
                # Cycle detected — break it
                node.parent = None
                if node in parent_node.children:
                    parent_node.children.remove(node)
                continue

            node.cost += delta
            visited.add(node)
            queue.extend(node.children)

    def _calc_new_cost(self, from_node: Node, to_node: Node) -> float:
        """
        Calculate the cost of moving from from_node to to_node.
        Includes a curvature penalty to encourage straighter paths.
        """
        path = to_node.path
        if len(path) > 0:
            # Aggiunge la distanza tra il nodo parent e il  punto della traiettoria simulata
            path_cost = math.hypot(from_node.x - path[0][0], from_node.y - path[0][1])
        else:
            path_cost = math.hypot(from_node.x - to_node.x, from_node.y - to_node.y)

        # Centerline curvature Penalty: Heavily penalize deviating from the track center
        cl_penalty = 0.0
        if self._cl_tree is not None:
            # KD-tree query O(log n) instead of linear scan
            _, ind_near_center_pt = self._cl_tree.query([from_node.x, from_node.y])

            if ind_near_center_pt == len(self.centerline) - 1:
                ind_near_center_pt -= 1

            p_cl1 = self.centerline[ind_near_center_pt]
            p_cl2 = self.centerline[ind_near_center_pt + 1]

            # Centerline vector (from point i to i+1)
            v_cl_x = p_cl2[0] - p_cl1[0]
            v_cl_y = p_cl2[1] - p_cl1[1]

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

        # ---THETA BETWEEN NODES PENALTY ---
        # Kinematic Ratio with
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        step_len_sq = dx*dx + dy*dy
        if step_len_sq > 1e-10:
            delta_theta = abs(normalize_angle(to_node.theta - from_node.theta))
            # ratio^2 ~= ((delta_theta * wheelbase) / (step_len * max_steering))^2
            num = delta_theta * self.wheelbase
            den = self.max_steering_angle
            steer_ratio_sq = (num * num) / (step_len_sq * den * den)
            
            heading_penalty = steer_ratio_sq #* HEADING_WEIGHT
        else:
            heading_penalty = 0.0

        return from_node.cost + path_cost + cl_penalty + heading_penalty

    def _get_near_nodes(self, new_node: Node) -> List[int]:
        """
        Find all nodes in the tree within the RRT* optimal radius.
        Uses the asymptotically optimal formula: r = γ · (log(n)/n)^(1/d)
        capped to a maximum and floored to a minimum for practical effectiveness.
        """
        n = len(self.node_list)
        if n <= 1:
            return []

        # RRT* optimal radius formula (d=2 dimensions)
        gamma = self.step_size * 10.0    # tuning constant
        radius = min(
            gamma * (math.log(n) / n) ** 0.5,   # optimal shrinking radius
            self.step_size * 5.0                  # maximum cap
        )
        # Minimum radius to ensure at least some rewiring
        radius = max(radius, self.step_size * 1.5)

        # NumPy vectorized radius query
        r_sq = radius * radius
        dists_sq = (self.node_coords[:n, 0] - new_node.x)**2 + (self.node_coords[:n, 1] - new_node.y)**2
        near_indices = np.where(dists_sq < r_sq)[0].tolist()

        return near_indices

    def _get_nearest_node_index(self, node_list: List[Node], rnd_node: Node) -> int:
        """
        Find the index of the nearest node in the tree to the sampled point.
        """
        n = len(node_list)
        dists_sq = (self.node_coords[:n, 0] - rnd_node.x)**2 + (self.node_coords[:n, 1] - rnd_node.y)**2
        return np.argmin(dists_sq)

    def _choose_parent(self, new_node: Node, near_node_indices: List[int]) -> Node:
        """
        Phase A of Rewiring: Find the best parent for the new_node among its near neighbors
        to minimize the cost to reach new_node.
        """
        if not near_node_indices:
            return new_node

        best_cost = inf
        best_parent = None
        best_path = []

        for i in near_node_indices:
            near_node = self.node_list[i]

            # Simulate steering from near node to new node
            simulated_node = self._steer(near_node, new_node, self.step_size)

            if simulated_node is None:
                continue

            # Calcola il potenziale costo se passassimo da questo vicino
            cost = self._calc_new_cost(near_node, simulated_node)

            if cost < best_cost:
                # Collision check solo se il costo è migliorativo
                if self.collision_checker.is_path_free(simulated_node.path):
                    best_cost = cost
                    best_parent = near_node
                    best_path = simulated_node.path

        # Se abbiamo trovato un padre migliore rispetto a quello originale assegnato in plan()
        if best_parent is not None:
            # Remove from old parent's children if applicable
            if new_node.parent is not None and new_node in new_node.parent.children:
                new_node.parent.children.remove(new_node)
            new_node.parent = best_parent
            new_node.cost = best_cost
            new_node.path = best_path

        return new_node

    def _rewire(self, new_node: Node, near_node_indices: List[int]):
        """
        Rewire nearby nodes through new_node if doing so reduces cost.
        """
        for i in near_node_indices:
            near_node = self.node_list[i]

            # Avoid rewiring the parent of new_node or the start node
            if near_node is new_node or near_node is new_node.parent or near_node is self.start:
                continue

            # Calculate distance to near_node
            dist = math.hypot(near_node.x - new_node.x, near_node.y - new_node.y)

            simulated_node = self._steer(new_node, near_node, dist)

            # Check if simulated_node actually reaches near the target node
            if math.hypot(simulated_node.x - near_node.x, simulated_node.y - near_node.y) > 0.2:
                continue

            # Calculate potential cost through new_node
            new_cost = self._calc_new_cost(new_node, simulated_node)

            if near_node.cost > new_cost:
                # Collision check
                if self.collision_checker.is_path_free(simulated_node.path):
                    # Remove near_node from its current parent's children list
                    if near_node.parent is not None and near_node in near_node.parent.children:
                        near_node.parent.children.remove(near_node)

                    # Update parent, cost, path
                    old_cost = near_node.cost
                    near_node.parent = new_node
                    near_node.cost = new_cost
                    near_node.path = simulated_node.path
                    # Add near_node to new_node's children
                    new_node.children.append(near_node)

                    # Propagate cost change to all descendants of near_node
                    self._propagate_cost_to_children(near_node, old_cost, new_cost)

    def _extract_path(self, last_node: Node) -> Tuple[List[Tuple[float, float, float]], float]:
        """
        Extract the full dense path from root to last_node by tracing parent chain
        and concatenating all intermediate kinematic paths.

        Returns:
            Tuple of (dense_path, cost) where dense_path is a list of (x, y, theta).
        """
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
        # Opzione 1: Percorso grezzo puro (nessuno smoothing)
        #return dense, last_node.cost

        # Opzione 2 (Attiva): RRT* + B-spline
        smoothed_path = self.Bspline(dense)
        return smoothed_path, last_node.cost

    def Bspline(self, path_pts: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
        """
        Performs B-spline interpolation using chord length parametrization.
        """
        if len(path_pts) < 2:
            return path_pts

        # Filter out consecutive duplicate points to prevent division by zero in splprep
        # Optimized to avoid math.hypot square root calls
        unique_pts = []
        for p in path_pts:
            if not unique_pts:
                unique_pts.append(p)
            else:
                last = unique_pts[-1]
                dx = p[0] - last[0]
                dy = p[1] - last[1]
                if dx * dx + dy * dy > 1e-8:
                    unique_pts.append(p)

        if len(unique_pts) < 2:
            return unique_pts

        x = [p[0] for p in unique_pts]
        y = [p[1] for p in unique_pts]

        # Calculate chord length parametrization (equations 8-11)
        dists = [0.0]
        for i in range(1, len(unique_pts)):
            d = math.hypot(x[i] - x[i-1], y[i] - y[i-1])
            dists.append(dists[-1] + d)

        total_length = dists[-1]
        if total_length < 1e-5:
            return unique_pts

        # Normalize parameters to [0, 1]
        u = np.array(dists) / total_length

        # Fit B-spline (Paul Dierckx's method)
        k = min(3, len(unique_pts) - 1)

        try:
            tck, u_evaluated = splprep([x, y], u=u, s=len(x) * 0.5, k=k)

            # Generate dense points along the curve (5cm intervals)
            num_points = max(10, int(total_length / 1)) # between 1 - 2
            u_new = np.linspace(0.0, 1.0, num_points)

            # Evaluate curve coordinates and first derivatives
            out_x, out_y = splev(u_new, tck)
            dx, dy = splev(u_new, tck, der=1)

            thetas = np.arctan2(dy, dx)
            smoothed_path = list(zip(out_x.tolist(), out_y.tolist(), thetas.tolist()))

            return smoothed_path
        except Exception:
            return unique_pts
