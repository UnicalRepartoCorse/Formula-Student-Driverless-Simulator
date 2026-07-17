import math
from math import inf
from typing import List, Tuple, Optional
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.spatial import Delaunay
from scipy.optimize import minimize_scalar

from .utils import dist_sq


def _ccw(P, Q, R):
    """Counter-clockwise check (helper function defined at module level for speed)."""
    return (Q[0] - P[0]) * (R[1] - P[1]) - (Q[1] - P[1]) * (R[0] - P[0])


class CollisionChecker:
    """
    Collision detection between sampled RRT* nodes and the track boundaries.

    Strategies
    ----------
    radial     : point must be farther than `cone_radius` from every cone. Just for testing
    boundaries : point must lie inside the corridor defined by the blue (left)
                 and yellow (right) spline boundaries.
    """

    def __init__(self, strategy: str = "boundaries", cone_radius: float = 0.9):
        """
        Parameters
        ----------
        strategy        : 'radial' or 'boundaries'
        cone_radius     : minimum safe distance from a cone centre [m]
        """
        self.strategy = strategy
        self.cone_radius = cone_radius
        self.cone_radius_sq = cone_radius * cone_radius

        self.blue_cones:   List[Tuple[float, float]] = []
        self.yellow_cones: List[Tuple[float, float]] = []
        self.orange_cones: List[Tuple[float, float]] = []

        # Dense boundary polylines (Nx2 numpy arrays) built from splines
        self._blue_pts:   Optional[np.ndarray] = None
        self._yellow_pts: Optional[np.ndarray] = None
        self._blue_tangents: Optional[np.ndarray] = None
        self._yellow_tangents: Optional[np.ndarray] = None
        self._blue_tck = None
        self._yellow_tck = None
        self._cones_array: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # UTILS
    # ------------------------------------------------------------------

    def update_cones(self,
                     blue_cones:   List[Tuple[float, float]],
                     yellow_cones: List[Tuple[float, float]],
                     orange_cones: List[Tuple[float, float]]):
        """Ingest new cone observations and rebuild internal structures."""
        self.blue_cones   = blue_cones
        self.yellow_cones = yellow_cones
        self.orange_cones = orange_cones

        all_cones = blue_cones + yellow_cones + orange_cones
        self._cones_array = np.array(all_cones) if all_cones else None

        return self._build_boundaries()

    def is_point_free(self, x: float, y: float) -> bool:
        """Return True if (x, y) is a collision-free configuration."""
        if self.strategy == "radial":
            return self._radial_check_point(x, y)

        if self.strategy == "boundaries":
            # Hard check: stay away from every cone physically
            if not self._radial_check_point(x, y):
                return False
            # Soft check: point must be inside the track corridor
            if self._blue_pts is not None and self._yellow_pts is not None:
                return self._point_in_track(x, y)
            return True

        raise NotImplementedError(f"Unknown strategy: {self.strategy}")

    def is_path_free(self, path: List[Tuple[float, float, float]]) -> bool:
        """
        Return True if every point and every segment of `path` is collision-free.
        """
        if not path:
            return True

        for pt in path:
            if not self.is_point_free(pt[0], pt[1]):
                return False

        if self.strategy == "boundaries" and self._blue_pts is not None and self._yellow_pts is not None:
            for i in range(len(path) - 1):
                if self._segment_crosses_boundaries(path[i], path[i + 1]):
                    return False
        return True

    def check_side(self, target_pt: Tuple[float, float], pts: np.ndarray, tangents: Optional[np.ndarray] = None) -> str:
        """
        Check if target_pt = (xp, yp) is to the left or right of the spline defined by its dense points.
        Returns: 'left', 'right', or 'on_line'
        """
        if pts is None or len(pts) < 2:
            return "on_line"
        
        # 1. Find closest point index using vectorized NumPy (squared distance) - optimized to avoid 2D allocation
        dists_sq = (pts[:, 0] - target_pt[0])**2 + (pts[:, 1] - target_pt[1])**2
        idx = np.argmin(dists_sq)
        
        # 2. Retrieve local direction (tangent) of the polyline at that index
        if tangents is not None and len(tangents) > idx:
            tx, ty = tangents[idx]
        else:
            # Fallback if tangents are not precomputed
            if idx == 0:
                tx = pts[1][0] - pts[0][0]
                ty = pts[1][1] - pts[0][1]
            elif idx == len(pts) - 1:
                tx = pts[-1][0] - pts[-2][0]
                ty = pts[-1][1] - pts[-2][1]
            else:
                tx = pts[idx + 1][0] - pts[idx - 1][0]
                ty = pts[idx + 1][1] - pts[idx - 1][1]
                
            length = math.hypot(tx, ty)
            if length > 0:
                tx /= length
                ty /= length
            else:
                tx, ty = 0.0, 0.0
            
        # 3. Vector from closest point to target point
        vx = target_pt[0] - pts[idx][0]
        vy = target_pt[1] - pts[idx][1]
        
        # 4. 2D cross product
        cross_product = tx * vy - ty * vx
        epsilon = 1e-5
        
        if cross_product > epsilon:
            return "left"
        elif cross_product < -epsilon:
            return "right"
        else:
            return "on_line"

    # ------------------------------------------------------------------
    # Boundary construction
    # ------------------------------------------------------------------

    def _compute_tangents(self, pts: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Precompute polyline tangents for fast lookup."""
        if pts is None or len(pts) < 2:
            return None
        n = len(pts)
        tangents = np.zeros((n, 2))
        tangents[0] = pts[1] - pts[0]
        tangents[-1] = pts[-1] - pts[-2]
        if n > 2:
            tangents[1:-1] = pts[2:] - pts[:-2]
        
        norms = np.hypot(tangents[:, 0], tangents[:, 1])
        valid = norms > 1e-9
        tangents[valid] /= norms[valid][:, np.newaxis]
        return tangents

    def _build_boundaries(self):
        if len(self.blue_cones) >= 2:
            #TODO DA RIMUOVERE POICHE RICEVO CONI GIA ORDNATI
            blue_sorted = sorted(self.blue_cones, key=lambda p: p[0])
            self._blue_pts, self._blue_tck = self.get_spline(np.array(blue_sorted))
        elif len(self.blue_cones) == 1:
            self._blue_pts = np.array(self.blue_cones)
        else:
            self._blue_pts = None

        if len(self.yellow_cones) >= 2:
            #TODO DA RIMUOVERE POICHE RICEVO CONI GIA ORDNATI
            yellow_sorted = sorted(self.yellow_cones, key=lambda p: p[0])
            self._yellow_pts, self._yellow_tck = self.get_spline(np.array(yellow_sorted))
        elif len(self.yellow_cones) == 1:
            self._yellow_pts = np.array(self.yellow_cones)
        else:
            self._yellow_pts = None

        # Precompute tangents for fast boundary side checks
        self._blue_tangents = self._compute_tangents(self._blue_pts)
        self._yellow_tangents = self._compute_tangents(self._yellow_pts)

        return self._blue_pts, self._yellow_pts

    def _intersect(self, A, B, C, D) -> bool:
        """
        Verifica se il segmento AB interseca il segmento CD.
        """
        # AABB (Bounding Box) check per rigetto rapido
        if (max(A[0], B[0]) < min(C[0], D[0]) or
            min(A[0], B[0]) > max(C[0], D[0]) or
            max(A[1], B[1]) < min(C[1], D[1]) or
            min(A[1], B[1]) > max(C[1], D[1])):
            return False

        return (_ccw(A, B, C) * _ccw(A, B, D) <= 0) and (_ccw(C, D, A) * _ccw(C, D, B) <= 0)

    def _segment_intersects_polyline(self, A, B, polyline) -> bool:
        """
        Ritorna True se il segmento AB interseca uno qualsiasi dei segmenti della polilinea.
        Utilizza una ricerca locale attorno al punto più vicino per massimizzare la velocità.
        """
        if polyline is None or len(polyline) < 2:
            return False
        
        # 1. Trova il punto della polilinea più vicino ad A usando NumPy vettorizzato
        dists_sq = (polyline[:, 0] - A[0])**2 + (polyline[:, 1] - A[1])**2
        idx = np.argmin(dists_sq)
        
        # 2. Controlla solo i segmenti adiacenti (finestra di ±5 indici)
        start = max(0, idx - 5)
        end = min(len(polyline) - 1, idx + 5)
        
        for i in range(start, end):
            if self._intersect(A, B, polyline[i], polyline[i+1]):
                return True
        return False

    def _segment_crosses_boundaries(self, from_node, to_node) -> bool:
        """
        Nodes are represented by (x, y, theta) or (x, y).
        Checks if the segment from_node -> to_node crosses any boundary.
        """
        A = (from_node[0], from_node[1])
        B = (to_node[0], to_node[1])

        if self._segment_intersects_polyline(A, B, self._blue_pts):
            return True
        if self._segment_intersects_polyline(A, B, self._yellow_pts):
            return True

        return False

    def _point_in_track(self, x: float, y: float) -> bool:
        return (self.check_side((x, y), self._blue_pts, self._blue_tangents) == "right" and 
                self.check_side((x, y), self._yellow_pts, self._yellow_tangents) == "left")

    # ------------------------------------------------------------------
    # Radial check
    # ------------------------------------------------------------------

    def _radial_check_point(self, x: float, y: float) -> bool:
        if self._cones_array is None or len(self._cones_array) == 0:
            return True
        dists_sq = (self._cones_array[:, 0] - x)**2 + (self._cones_array[:, 1] - y)**2
        return not np.any(dists_sq < self.cone_radius_sq)

    def get_spline(self, points):
        res = 0.5
        try:
            # Parametric spline fit
            tck, u = splprep([points[:, 0], points[:, 1]], s=len(points) * 0.5, k=2)
            # Estimate total arc length
            diffs = np.diff(points, axis=0)
            arc = np.sum(np.hypot(diffs[:, 0], diffs[:, 1]))
            n_samples = max(int(arc / res), len(points))
            u_new = np.linspace(0, 1, n_samples)
            sx, sy = splev(u_new, tck)
            return np.column_stack([sx, sy]), tck
        except Exception as e:
            print(f"Spline failed, using polyline: {e}")
            return points, None

    def xy_limit(self):
        pass