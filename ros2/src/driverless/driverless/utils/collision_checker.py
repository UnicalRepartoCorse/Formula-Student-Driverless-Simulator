import math
from math import inf
from typing import List, Tuple, Optional
import numpy as np
from scipy.interpolate import splprep, splev


class CollisionChecker:
    """
    Collision detection between sampled RRT* nodes and the track boundaries.

    Strategies
    ----------
    radial     : point must be farther than `cone_radius` from every cone.
    boundaries : point must lie inside the corridor defined by the blue (left)
                 and yellow (right) spline boundaries.
    """

    def __init__(self, strategy: str = "boundaries", cone_radius: float = 0.9,
                 spline_samples: int = 200):
        """
        Parameters
        ----------
        strategy        : 'radial' or 'boundaries'
        cone_radius     : minimum safe distance from a cone centre [m]
        spline_samples  : number of points used to densely sample each boundary
                          spline (higher → more accurate curve representation)
        """
        self.strategy = strategy
        self.cone_radius = cone_radius
        self.spline_samples = spline_samples

        self.blue_cones:   List[Tuple[float, float]] = []
        self.yellow_cones: List[Tuple[float, float]] = []
        self.orange_cones: List[Tuple[float, float]] = []

        # Dense boundary polylines (Nx2 numpy arrays) built from splines
        self._blue_pts:   Optional[np.ndarray] = None
        self._yellow_pts: Optional[np.ndarray] = None

        # Closed track polygon for point-in-polygon test (Nx2)
        self._track_polygon: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_cones(self,
                     blue_cones:   List[Tuple[float, float]],
                     yellow_cones: List[Tuple[float, float]],
                     orange_cones: List[Tuple[float, float]]):
        """Ingest new cone observations and rebuild internal structures."""
        self.blue_cones   = blue_cones
        self.yellow_cones = yellow_cones
        self.orange_cones = orange_cones

        if self.strategy == "boundaries":
            self._build_boundaries()

    def is_point_free(self, x: float, y: float) -> bool:
        """Return True if (x, y) is a collision-free configuration."""
        if self.strategy == "radial":
            return self._radial_check_point(x, y)

        if self.strategy == "boundaries":
            # Hard check: stay away from every cone physically
            if not self._radial_check_point(x, y):
                return False
            # Soft check: point must be inside the track corridor
            if self._track_polygon is not None:
                return self._point_in_polygon(x, y, self._track_polygon)
            return True

        raise NotImplementedError(f"Unknown strategy: {self.strategy}")

    def is_path_free(self, path: List[Tuple[float, float]]) -> bool:
        """
        Return True if every point and every segment of `path` is collision-free.
        """
        if not path:
            return True

        for pt in path:
            if not self.is_point_free(pt[0], pt[1]):
                return False

        if self.strategy == "boundaries" and \
                self._blue_pts is not None and self._yellow_pts is not None:
            for i in range(len(path) - 1):
                if self._segment_crosses_boundaries(path[i], path[i + 1]):
                    return False

        return True

    # ------------------------------------------------------------------
    # Boundary construction
    # ------------------------------------------------------------------

    def _build_boundaries(self):
        """Order cones, fit splines, sample densely, build track polygon.

        Orange cones mark the start line and are used as anchors for both
        splines, ensuring the track polygon correctly encloses the vehicle
        starting position at (0, 0).
        """
        if len(self.blue_cones) < 2 or len(self.yellow_cones) < 2:
            self._blue_pts = None
            self._yellow_pts = None
            self._track_polygon = None
            return

        blue_ord   = self._nearest_neighbour_sort(self.blue_cones)
        yellow_ord = self._nearest_neighbour_sort(self.yellow_cones)

        # --- Anchor splines at the start line using orange cones ---
        # Orange cones sit at the entry gate. Split them by Y sign:
        #   Y > 0  →  left side  →  prepend to blue boundary
        #   Y ≤ 0  →  right side →  prepend to yellow boundary
        # If an orange cone sits exactly on the centreline (Y≈0) assign it
        # to the side whose first ordered cone it is closest to.
        if self.orange_cones:
            left_anchors  = []
            right_anchors = []
            for oc in self.orange_cones:
                ox, oy = oc
                if oy > 0:
                    left_anchors.append(oc)
                elif oy < 0:
                    right_anchors.append(oc)
                else:
                    # Y == 0: assign to nearest boundary
                    d_blue   = math.hypot(ox - blue_ord[0][0],   oy - blue_ord[0][1])
                    d_yellow = math.hypot(ox - yellow_ord[0][0], oy - yellow_ord[0][1])
                    (left_anchors if d_blue <= d_yellow else right_anchors).append(oc)

            # Sort anchors by distance from origin so the nearest gate-cone
            # becomes the first vertex of the spline.
            left_anchors.sort( key=lambda c: math.hypot(c[0], c[1]))
            right_anchors.sort(key=lambda c: math.hypot(c[0], c[1]))

            # Prepend: anchor → ordered boundary cones
            blue_ord   = left_anchors  + blue_ord
            yellow_ord = right_anchors + yellow_ord

        self._blue_pts   = self._fit_and_sample(blue_ord)
        self._yellow_pts = self._fit_and_sample(yellow_ord)

        if self._blue_pts is not None and self._yellow_pts is not None:
            # Close the polygon: blue forward + yellow reversed
            # This traces the track corridor as a single closed ring.
            self._track_polygon = np.vstack([
                self._blue_pts,
                self._yellow_pts[::-1]
            ])

    @staticmethod
    def _nearest_neighbour_sort(cones: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        Greedy nearest-neighbour ordering starting from the cone closest to
        the vehicle origin (0, 0) — i.e. the first cone in the vehicle's path.

        Works correctly in curves because it follows spatial proximity rather
        than a single axis, so the resulting order matches the physical track
        layout for any track shape.
        """
        if not cones:
            return []

        remaining = list(cones)

        # Start from the cone nearest to the vehicle
        # Usa: cono più avanti (max X) tra i N più vicini
        candidates = sorted(remaining, key=lambda c: math.hypot(c[0], c[1]))[:3]
        start_idx = remaining.index(max(candidates, key=lambda c: c[0]))
        ordered = [remaining.pop(start_idx)]

        while remaining:
            last = ordered[-1]
            nearest_idx = min(range(len(remaining)),
                              key=lambda i: math.hypot(remaining[i][0] - last[0],
                                                       remaining[i][1] - last[1]))
            ordered.append(remaining.pop(nearest_idx))

        return ordered

    def _fit_and_sample(self, ordered_cones: List[Tuple[float, float]]) -> Optional[np.ndarray]:
        """
        Fit a parametric cubic spline through `ordered_cones` and return a
        dense array of (x, y) samples.

        Parametric form — t is cumulative arc-length between cone positions —
        so the spline handles curves, hairpins and any non-monotonic shape
        without the y = f(x) limitation.
        """
        pts = np.array(ordered_cones, dtype=float)  # (N, 2)
        n = len(pts)

        if n < 2:
            return None

        # Polynomial degree: cubic when possible, lower for very few points
        k = min(3, n - 1)

        # Arc-length parameterisation avoids clustering artefacts in curves
        diffs = np.diff(pts, axis=0)
        seg_lengths = np.hypot(diffs[:, 0], diffs[:, 1])
        t = np.concatenate([[0.0], np.cumsum(seg_lengths)])
        t /= t[-1]  # normalise to [0, 1]

        try:
            # s=0  → interpolating spline (passes through every cone exactly)
            tck, _ = splprep([pts[:, 0], pts[:, 1]], u=t, s=0, k=k)
        except Exception:
            # Degenerate geometry — fall back to the raw cone positions
            return pts

        u_dense = np.linspace(0.0, 1.0, self.spline_samples)
        x_s, y_s = splev(u_dense, tck)
        return np.column_stack([x_s, y_s])  # (spline_samples, 2)

    # ------------------------------------------------------------------
    # Geometric helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _radial_check_point(x: float, y: float,
                            cones: Optional[List[Tuple[float, float]]] = None,
                            radius: float = 0.0) -> bool:
        """Reusable; called as a method too — see is_point_free."""
        # This is called as self._radial_check_point so we need instance access.
        raise RuntimeError("Use instance method")

    def _radial_check_point(self, x: float, y: float) -> bool:  # noqa: F811
        all_cones = self.blue_cones + self.yellow_cones + self.orange_cones
        for cx, cy in all_cones:
            if math.hypot(x - cx, y - cy) < self.cone_radius:
                return False
        return True

    @staticmethod
    def _point_in_polygon(x: float, y: float, polygon: np.ndarray) -> bool:
        """
        Ray-casting algorithm.
        Cast a horizontal ray from (x, y) to +∞ and count crossings with the
        polygon edges.  Odd count → inside.

        Robust for any convex or concave closed polygon.
        """
        n = len(polygon)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            # Edge crosses the horizontal ray at x?
            if ((yi > y) != (yj > y)) and \
                    (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
                inside = not inside
            j = i
        return inside

    def _segment_crosses_boundaries(self,
                                    p1: Tuple[float, float],
                                    p2: Tuple[float, float]) -> bool:
        """
        Return True if the segment p1→p2 intersects any edge of the dense
        blue or yellow boundary polylines.
        """
        for boundary in (self._blue_pts, self._yellow_pts):
            if boundary is None:
                continue
            for i in range(len(boundary) - 1):
                if self._segments_intersect(p1, p2,
                                            tuple(boundary[i]),
                                            tuple(boundary[i + 1])):
                    return True
        return False

    @staticmethod
    def _ccw(A, B, C) -> bool:
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    def _segments_intersect(self, A, B, C, D) -> bool:
        return (self._ccw(A, C, D) != self._ccw(B, C, D) and
                self._ccw(A, B, C) != self._ccw(A, B, D))