import math
import random
from math import inf
from typing import List, Tuple, Optional
from scipy.interpolate import splprep, splev

from driverless.utils.collision_checker import CollisionChecker

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
                 centerline: Optional[List[Tuple[float, float]]] = None):
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

        self.node_list = [self.start]
        self.sampled_points = []  # Memorizza tutti i punti campionati ad ogni passo
        self.sample_radius_centerline = sample_radius_centerline #m

    def _dist_to_goal_line(self, x, y):
        """
        Calculates the shortest distance from a point (x, y) to the goal line segment.
        """
        p = (x, y)
        a, b = self.goal_line

        # Vector segment AB
        ax, ay = a
        bx, by = b
        dx = bx - ax
        dy = by - ay

        # If segment is just a point
        if dx == 0 and dy == 0:
            return math.hypot(x - ax, y - ay)

        # Projection factor t
        t = ((x - ax) * dx + (y - ay) * dy) / (dx * dx + dy * dy)
        t = max(0.0, min(1.0, t)) # Clamp to segment range

        # Closest point on segment
        proj_x = ax + t * dx
        proj_y = ay + t * dy

        return math.hypot(x - proj_x, y - proj_y)

    def plan(self) -> Optional[List[Tuple[float, float, float]]]:
        """
        Execute the RRT* planning algorithm.
        
        Returns:
            List of (x, y, theta) representing the path, or None if no path found.
        """
        first_goal_idx = None
        for i in range(0, self.max_iter):
            # Limite dinamico: interrompiamo 100 iterazioni dopo il primo goal trovato,
            # o a un massimo di 300 iterazioni totali per rimanere entro 20ms in Python.
            #if first_goal_idx is not None and (i - first_goal_idx >= 100):
            #    break
            #if i >= 300:
            #    break

            sample = self._sample_free_space() # Node: x, y, theta
            self.sampled_points.append((sample.x, sample.y))

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

            self._rewire(new_node, nearest_indeces)

            # Segnala se abbiamo raggiunto il traguardo
            if new_node.dist_to_goal <= self.step_size and first_goal_idx is None:
                first_goal_idx = i

        # Selezione del percorso migliore
        goal_nodes = [node for node in self.node_list if node.dist_to_goal <= self.step_size]

        if goal_nodes:
            # Tra quelli al traguardo, scegliamo quello con il costo minore (più corto e dritto)
            best_node = min(goal_nodes, key=lambda n: n.cost)
        else:
            # Fallback: se nessuno ha raggiunto la goal line, prendiamo il più vicino
            best_node = min(self.node_list, key=lambda n: n.dist_to_goal)

        if best_node is not None:
            return self._extract_path(best_node)
        else:
            return None

    def _sample_free_space(self) -> Node:
        """
        Randomly sample a point (x, y, theta).
        If a centerline is provided, strictly sample within a 1.5m radius 
        of a randomly chosen point on that centerline.
        Otherwise, sample uniformly within the bounds.
        """
        if self.centerline and len(self.centerline) > 0:
            # Pick a random reference point on the centerline
            ref_pt = random.choice(self.centerline)

            # Sample within a circle of radius R (1.5m) around the reference point

            r = self.sample_radius_centerline * math.sqrt(random.uniform(0, 1))
            alpha = random.uniform(0, 2 * math.pi)

            x = ref_pt[0] + r * math.cos(alpha)
            y = ref_pt[1] + r * math.sin(alpha)
            theta = random.uniform(-math.pi, math.pi)
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

        # Distance to target
        dx = target.x - from_node.x
        dy = target.y - from_node.y

        dist_to_target = math.hypot(dx, dy)

        if dist_to_target < 1e-6:
            return None

        # Limit expansion distance
        move_dist = min(self.step_size, dist_to_target)

        # Spatial integration resolution

        steps = max(1, int(move_dist / self.step_size))

        # Recompute dt to keep exact move distance
        v = 1.0
        dt = move_dist / (steps * v)

        # Initial state
        x = from_node.x
        y = from_node.y
        theta = from_node.theta

        new_node = Node(x, y, theta)

        # Simulate bicycle model
        for _ in range(steps):

            # Direction toward target
            target_angle = math.atan2(target.y - y, target.x - x)

            # Heading error normalized [-pi, pi]
            heading_error = target_angle - theta
            heading_error = (heading_error + math.pi) % (2 * math.pi) - math.pi

            # Pure Pursuit inspired steering law
            steering_angle = math.atan2(
                2.0 * self.wheelbase * math.sin(heading_error),
                move_dist
            )

            # Respect steering limits
            steering_angle = max(
                -self.max_steering_angle,
                min(self.max_steering_angle, steering_angle)
            )

            # Bicycle model integration
            x += v * math.cos(theta) * dt
            y += v * math.sin(theta) * dt

            theta += (
                             v / self.wheelbase
                     ) * math.tan(steering_angle) * dt

            # Normalize theta
            theta = (theta + math.pi) % (2 * math.pi) - math.pi

            new_node.path.append((x, y,theta))

        # Final state
        new_node.x = x
        new_node.y = y
        new_node.theta = theta

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
        and a strong centerline penalty to keep the path centered.
        """
        path_cost = 0.0
        path = to_node.path
        if len(path) > 0:
            # Aggiunge la distanza tra il nodo parent e il primo punto della traiettoria simulata
            path_cost += math.hypot(from_node.x - path[0][0], from_node.y - path[0][1])
            # Aggiunge la distanza tra tutti i punti consecutivi
            for i in range(len(path)-1):
                p1 = path[i]
                p2 = path[i+1]
                path_cost += math.hypot(p1[0] - p2[0], p1[1] - p2[1])
        else:
            path_cost += math.hypot(from_node.x - to_node.x, from_node.y - to_node.y)


        # Curvature Penalty: Penalize large changes in heading
        delta_theta = abs(to_node.theta - from_node.theta)
        delta_theta = min(delta_theta, 2*math.pi - delta_theta) # Shortest angular distance
        curvature_penalty = 3 * delta_theta # K_theta = 1.5

        ''' SUS DA VALUTARE

        # Centerline Penalty: Heavily penalize deviating from the track center
        cl_penalty = 0.0
        if self.centerline and len(self.centerline) > 0:
            min_d = inf
            for cx, cy in self.centerline:
                d = math.hypot(to_node.x - cx, to_node.y - cy)
                if d < min_d:
                    min_d = d
            # The further from the centerline, the exponentially higher the cost
            cl_penalty = 50.0 * (min_d ** 2)
        
        '''

        return from_node.cost + path_cost + curvature_penalty #+ cl_penalty

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
            simulated_node = self._steer(near_node, new_node)

            if simulated_node is None:
                continue

            # Se la strada è libera da ostacoli
            if self.collision_checker.is_path_free(simulated_node.path):
                # Calcola il potenziale costo se passassimo da questo vicino
                cost = self._calc_new_cost(near_node, simulated_node)

                if cost < best_cost:
                    best_cost = cost
                    best_parent = near_node
                    best_path = simulated_node.path

        # Se abbiamo trovato un padre migliore rispetto a quello originale assegnato in plan()
        if best_parent is not None:
            new_node.parent = best_parent
            new_node.cost = best_cost
            new_node.path = best_path

        return new_node

    def _rewire(self, new_node: Node, near_node_indices: List[int]):
        """
        Rewire nearby nodes through new_node if doing so reduces cost.

        This implementation:
        - checks kinematic feasibility
        - checks collision-free path
        - verifies actual geometric reachability
        - avoids cycles
        - propagates updated costs
        """

        for i in near_node_indices:

            near_node = self.node_list[i]

            # Avoid self rewiring
            if near_node is new_node:
                continue

            # Avoid trivial cycles
            if near_node.parent is new_node:
                continue

            # Generate feasible trajectory
            simulated_node = self._steer(
                new_node,
                near_node
            )

            if simulated_node is None:
                continue

            # Collision check
            if not self.collision_checker.is_path_free(simulated_node.path):
                continue

            # IMPORTANT:
            # verify we actually reached the target node
            reach_dist = math.hypot(
                simulated_node.x - near_node.x,
                simulated_node.y - near_node.y
            )

            # Tolerance threshold
            if reach_dist > self.step_size * 2:
                continue

            # Check for deeper cycles before rewiring
            # We must not set near_node.parent = new_node if new_node is a descendant of near_node
            is_cycle = False
            curr = new_node
            while curr is not None:
                if curr is near_node:
                    is_cycle = True
                    break
                curr = curr.parent

            if is_cycle:
                continue

            # Compute potential new cost
            new_cost = self._calc_new_cost(new_node, simulated_node)

            # Rewire only if cost improves
            if new_cost < near_node.cost:

                old_cost = near_node.cost

                near_node.parent = new_node
                near_node.cost = new_cost
                near_node.path = simulated_node.path

                # Propagate cost improvement to descendants
                self._propagate_cost_to_children(
                    near_node,
                    old_cost,
                    new_cost
                )

    def _extract_path(self, last_node: Node) -> List[Tuple[float, float, float]]:
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
        #return dense

        # Opzione 2 (Disattivata): Percorso levigato con spline cubica (campionamento equispaziato ad arco)
        return self._smooth_path(dense)

    def _smooth_path(
            self, path: List[Tuple[float, float, float]]
    ) -> List[Tuple[float, float, float]]:
        import numpy as np
        from scipy.interpolate import splprep, splev

        if len(path) < 4:
            return path

        pts = np.array([(p[0], p[1]) for p in path])

        try:
            # Fit parametrico della spline
            tck, u = splprep([pts[:, 0], pts[:, 1]], s=len(path) * 0.5, k=3)
        except Exception:
            return path

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
            return path

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
        return smoothed