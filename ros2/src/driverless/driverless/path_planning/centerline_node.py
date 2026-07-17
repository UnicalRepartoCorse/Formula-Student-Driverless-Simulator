#!/usr/bin/env python3
"""
Centerline Calculator Node for Formula Student Driverless (EUFS Simulation).

Computes the track centerline from blue/yellow cone positions using Delaunay
triangulation, robust edge filtering, direction-aware ordering, spline
smoothing, and temporal stabilization.

Author: Rewritten and improved from original delaunay_node.py
"""

import math
import numpy as np
from scipy.spatial import Delaunay
from scipy.interpolate import splprep, splev

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from fs_msgs.msg import Track


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def quat_to_yaw(q):
    """
    Converte un quaternione (geometry_msgs/Quaternion) nel corrispondente angolo
    di yaw (rotazione attorno all'asse Z verticale).

    Utilizza la formula standard per estrarre la componente yaw dalla
    rappresentazione quaternionica di un orientamento 3D. Per veicoli planari
    come quelli di Formula Student, yaw è l'unico angolo rilevante.

    Parametri:
        q: geometry_msgs/Quaternion con campi x, y, z, w.

    Ritorna:
        float: angolo yaw in radianti, nell'intervallo [-pi, pi].
    """
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def angle_diff(a, b):
    """
    Calcola la differenza angolare più breve tra due angoli, con segno.

    Gestisce correttamente il wrap-around a ±pi. Ad esempio, la differenza
    tra +170° e -170° risulta 20° (non 340°). Viene usata per confrontare
    la direzione di heading del veicolo con la direzione verso un cono
    o un midpoint, evitando errori dovuti al passaggio da +pi a -pi.

    Parametri:
        a (float): primo angolo in radianti.
        b (float): secondo angolo in radianti.

    Ritorna:
        float: differenza (a - b) normalizzata nell'intervallo [-pi, pi].
    """
    d = a - b
    return (d + math.pi) % (2.0 * math.pi) - math.pi


def rgba(r, g, b, a=1.0):
    """
    Crea un oggetto std_msgs/ColorRGBA con i valori specificati.

    Helper di comodità per evitare di ripetere la costruzione manuale
    del messaggio colore ogni volta che si crea un marker RViz.

    Parametri:
        r (float): componente rossa [0.0 - 1.0].
        g (float): componente verde [0.0 - 1.0].
        b (float): componente blu [0.0 - 1.0].
        a (float): opacità [0.0 - 1.0], default 1.0 (completamente opaco).

    Ritorna:
        ColorRGBA: messaggio ROS pronto per essere assegnato a marker.color.
    """
    c = ColorRGBA()
    c.r, c.g, c.b, c.a = float(r), float(g), float(b), float(a)
    return c


# ---------------------------------------------------------------------------
# Main node
# ---------------------------------------------------------------------------

class CenterlineNode(Node):
    """
    Nodo ROS 2 che calcola la centerline (linea centrale) di una pista
    Formula Student Driverless a partire dalle posizioni dei coni blu e gialli.

    Pipeline di elaborazione (eseguita periodicamente dal timer):
        1. Pre-filtro dei coni per distanza e campo visivo
        2. Triangolazione di Delaunay sulle posizioni dei coni filtrati
        3. Selezione degli edge validi (solo blu↔giallo, entro soglie di larghezza)
        4. Calcolo dei midpoint con deduplicazione
        5. Ordinamento direzionale dei midpoint (greedy con penalità angolare)
        6. Smoothing con spline cubica e ricampionamento a passo costante
        7. Fusione temporale con la centerline precedente per stabilità
        8. Pubblicazione del Path e dei marker di debug
    """

    # Codici colore usati internamente per identificare i coni
    BLUE = 0      # Coni blu: delimitano il lato sinistro della pista
    YELLOW = 1    # Coni gialli: delimitano il lato destro della pista

    def __init__(self):
        """
        Inizializza il nodo: dichiara parametri, crea publisher/subscriber
        e avvia il timer periodico che esegue la pipeline di calcolo.
        """
        super().__init__('centerline_calculator')

        # -- Declare all ROS 2 parameters --
        self._declare_params()

        # -- State --
        self._cones: list[tuple[float, float, int]] = []
        self._cones_frame: str = 'base_footprint'
        self._car_x: float = 0.0
        self._car_y: float = 0.0
        self._car_yaw: float = 0.0
        self._prev_centerline: np.ndarray | None = None  # (N, 2)

        # -- Publishers --
        self._pub_path = self.create_publisher(
            Path, self._p('centerline_topic'), 10)
        self._pub_cones = self.create_publisher(
            MarkerArray, self._p('marker_topic'), 10)
        self._pub_debug = self.create_publisher(
            MarkerArray, '/debug/centerline', 10)

        # -- Subscribers --
        cone_qos = QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(
            Track, self._p('cone_topic'),
            self._cb_cones, cone_qos)
        self.get_logger().info(
            f"Subscribed to cones on: {self._p('cone_topic')}")

        self.create_subscription(
            Odometry, self._p('odom_topic'), self._cb_odom, 10)

        # -- Timer --
        period = self._p('timer_period')
        self.create_timer(period, self._pipeline)
        self.get_logger().info(
            f"Centerline node started (period={period:.2f}s)")

    # -----------------------------------------------------------------------
    # Parameter helpers
    # -----------------------------------------------------------------------

    def _declare_params(self):
        """
        Dichiara tutti i parametri ROS 2 con i rispettivi valori di default.

        I parametri sono organizzati in categorie:
        - Topic: nomi dei topic di input/output
        - Timing: periodo di esecuzione della pipeline
        - Pre-filtro: raggio e campo visivo per filtrare i coni
        - Edge selection: soglie min/max della larghezza pista
        - Midpoints: distanza di merge per deduplicazione
        - Ordering: passo massimo e peso direzionale per l'ordinamento
        - Smoothing: abilitazione e risoluzione della spline
        - Temporale: fattore alpha per il blend con la centerline precedente
        - Debug: abilitazione dei marker di visualizzazione intermedi

        Tutti i parametri possono essere modificati a runtime con:
            ros2 param set /centerline_calculator <nome_parametro> <valore>
        """
        d = self.declare_parameter
        # Topics
        d('cone_topic', '/fsds/testing_only/track')
        d('odom_topic', '/fsds/testing_only/odom')
        d('centerline_topic', '/track/centerline')
        d('marker_topic', '/viz/cones')
        # Timing
        d('timer_period', 0.1)
        # Pre-filter
        d('local_cone_radius', 20.0)
        d('front_fov_deg', 270.0)
        d('use_forward_filter', True)
        # Edge selection
        d('track_width_min', 2.0)
        d('track_width_max', 5.5)
        # Midpoints
        d('midpoint_merge_distance', 0.5)
        # Ordering
        d('max_progression_step', 8.0)
        d('ordering_direction_weight', 0.4)
        # Smoothing
        d('smoothing_enabled', True)
        d('smoothing_resolution', 0.5)
        # Temporal
        d('temporal_filter_alpha', 0.5)
        # Debug
        d('debug_markers_enabled', True)

    def _p(self, name: str):
        """
        Scorciatoia per leggere il valore corrente di un parametro ROS 2.

        Evita di scrivere self.get_parameter(name).value ogni volta,
        rendendo il codice più compatto e leggibile.

        Parametri:
            name (str): nome del parametro dichiarato in _declare_params().

        Ritorna:
            Il valore corrente del parametro (tipo dipende dalla dichiarazione).
        """
        return self.get_parameter(name).value

    # -----------------------------------------------------------------------
    # Callbacks
    # -----------------------------------------------------------------------

    def _cb_odom(self, msg: Odometry):
        """
        Callback dell'odometria: aggiorna la posizione e orientamento del veicolo.

        Estrae posizione (x, y) e yaw dal messaggio nav_msgs/Odometry.
        Questi valori servono nel frame globale (map) per sapere dove si
        trova l'auto e filtrare i coni di conseguenza. Nel frame locale
        (base_footprint) non vengono usati perché l'auto è sempre all'origine.

        Parametri:
            msg (Odometry): messaggio odometria dal simulatore EUFS.
        """
        self._car_x = msg.pose.pose.position.x
        self._car_y = msg.pose.pose.position.y
        self._car_yaw = quat_to_yaw(msg.pose.pose.orientation)

    def _cb_cones(self, msg):
        """
        Callback dei coni: converte il messaggio Track nella lista interna.

        Riceve fs_msgs/Track che contiene un array di coni (track).
        Estrae solo i coni blu e gialli (quelli che definiscono i bordi della
        pista) e li salva come lista di tuple (x, y, colore).

        I coni arancioni (start/finish) vengono ignorati intenzionalmente
        perché non contribuiscono alla definizione della larghezza pista.

        Parametri:
            msg: fs_msgs/Track dal simulatore.
        """
        cones = []
        for c in msg.track:
            if c.color == self.BLUE:
                cones.append((c.location.x, c.location.y, self.BLUE))
            elif c.color == self.YELLOW:
                cones.append((c.location.x, c.location.y, self.YELLOW))
        # I coni arancioni (big_orange / orange) vengono ignorati intenzionalmente
        self._cones = cones
        self._cones_frame = 'fsds/map'

    # -----------------------------------------------------------------------
    # Main pipeline (called by timer)
    # -----------------------------------------------------------------------

    def _pipeline(self):
        """
        Pipeline principale di calcolo della centerline, chiamata dal timer.

        Esegue in sequenza tutti gli step dell'algoritmo:
        A) Determina la posa del veicolo nel frame dei coni
        B) Filtra i coni per distanza e FOV
        C) Esegue triangolazione di Delaunay ed estrae gli edge
        D) Valida gli edge (solo blu↔giallo, soglie distanza, best-match)
        E) Calcola i midpoint dagli edge validi e li deduplica
        F) Ordina i midpoint con algoritmo greedy direzionale
        G) Applica smoothing con spline cubica
        H) Fonde la nuova centerline con quella precedente (stabilità temporale)
        I) Pubblica il Path finale e i marker di visualizzazione

        Se in qualsiasi step i dati sono insufficienti (pochi coni, nessun
        edge valido, pochi midpoint), la pipeline esce con un log di debug.
        """
        car_x, car_y, car_yaw = self._get_vehicle_pose()

        # A. Pre-filter
        cones = self._filter_cones(self._cones, car_x, car_y, car_yaw)
        if len(cones) < 4:
            self.get_logger().debug("Too few cones for triangulation.")
            return

        points = np.array([[c[0], c[1]] for c in cones])
        colors = np.array([c[2] for c in cones])

        # B. Triangulation + edge extraction
        edges = self._triangulate_and_extract(points)
        if not edges:
            return

        # B. Edge validation
        valid_edges, candidate_edges = self._select_valid_edges(
            edges, points, colors, car_x, car_y, car_yaw)

        # C. Midpoints
        midpoints = self._compute_midpoints(valid_edges, points)
        if len(midpoints) < 2:
            self.get_logger().debug("Too few midpoints to build centerline.")
            return

        # D. Order
        ordered = self._order_midpoints(midpoints, car_x, car_y, car_yaw)
        if len(ordered) < 2:
            return

        # E. Smooth
        smoothed = self._smooth_centerline(ordered)

        # F. Temporal fusion
        fused = self._temporal_fuse(smoothed, car_x, car_y)
        self._prev_centerline = fused.copy()

        # Publish
        self._publish_path(fused)
        self._publish_cone_markers(cones)

        # H. Debug markers
        if self._p('debug_markers_enabled'):
            self._publish_debug_markers(
                points, candidate_edges, valid_edges,
                midpoints, ordered, fused)

    # -----------------------------------------------------------------------
    # A. Vehicle pose
    # -----------------------------------------------------------------------

    def _get_vehicle_pose(self) -> tuple[float, float, float]:
        """
        Restituisce la posa (x, y, yaw) del veicolo nel frame dei coni.

        La logica cambia in base al frame di riferimento:
        - Se i coni arrivano nel frame locale (base_footprint o base_link),
          l'auto è per definizione all'origine (0, 0) con yaw=0.
        - Se i coni arrivano nel frame globale (map), la posizione dell'auto
          viene presa dall'odometria (aggiornata dal callback _cb_odom).

        Ritorna:
            tuple (car_x, car_y, car_yaw): posizione e heading del veicolo.
        """
        if self._cones_frame in ('base_footprint', 'base_link'):
            return 0.0, 0.0, 0.0
        return self._car_x, self._car_y, self._car_yaw

    # -----------------------------------------------------------------------
    # A. Cone pre-filter
    # -----------------------------------------------------------------------

    def _filter_cones(self, cones, car_x, car_y, car_yaw):
        """
        Pre-filtra i coni mantenendo solo quelli rilevanti per il calcolo.

        Applica due filtri in cascata:
        1. FILTRO DI DISTANZA: scarta i coni oltre local_cone_radius metri
           dall'auto. Questo riduce il numero di coni nella triangolazione
           ed evita che Delaunay crei triangoli enormi con coni lontani.

        2. FILTRO DI CAMPO VISIVO (opzionale, attivato da use_forward_filter):
           scarta i coni che stanno fuori dal cono visivo frontale dell'auto,
           definito dall'angolo front_fov_deg centrato sulla direzione di
           marcia (car_yaw). I coni a meno di 0.5m dall'auto vengono sempre
           inclusi (evita di scartare coni direttamente sotto il veicolo).

           Esempio: con front_fov_deg=270°, vengono esclusi solo i coni
           nei 90° direttamente dietro l'auto.

        Parametri:
            cones: lista di tuple (x, y, colore) di tutti i coni ricevuti.
            car_x, car_y: posizione del veicolo nel frame dei coni.
            car_yaw: heading del veicolo in radianti.

        Ritorna:
            list: coni filtrati come lista di tuple (x, y, colore).
        """
        radius = self._p('local_cone_radius')
        use_fov = self._p('use_forward_filter')
        half_fov = math.radians(self._p('front_fov_deg') / 2.0)

        filtered = []
        for cx, cy, color in cones:
            dx, dy = cx - car_x, cy - car_y
            dist = math.hypot(dx, dy)
            if dist > radius:
                continue
            if use_fov and dist > 0.5:
                # Calcola l'angolo dal veicolo verso il cono e verifica
                # che rientri nel campo visivo (half_fov da ciascun lato)
                angle_to_cone = math.atan2(dy, dx)
                if abs(angle_diff(angle_to_cone, car_yaw)) > half_fov:
                    continue
            filtered.append((cx, cy, color))
        return filtered

    # -----------------------------------------------------------------------
    # B. Triangulation
    # -----------------------------------------------------------------------

    def _triangulate_and_extract(self, points: np.ndarray) -> set | None:
        """
        Esegue la triangolazione di Delaunay ed estrae gli edge unici.

        Delaunay divide il piano in triangoli usando le posizioni dei coni
        come vertici, in modo che nessun punto cada dentro il circumcerchio
        di alcun triangolo. Da ogni triangolo si estraggono i 3 lati (edge)
        come coppie di indici (i, j) ordinate per evitare duplicati.

        Parametri:
            points (np.ndarray): array Nx2 con le coordinate dei coni filtrati.

        Ritorna:
            set di tuple (i, j) con gli indici degli edge unici,
            oppure None se la triangolazione fallisce.
        """
        try:
            tri = Delaunay(points)
        except Exception as e:
            self.get_logger().warn(f"Delaunay failed: {e}")
            return None
        edges = set()
        for s in tri.simplices:
            edges.add((min(s[0], s[1]), max(s[0], s[1])))
            edges.add((min(s[1], s[2]), max(s[1], s[2])))
            edges.add((min(s[0], s[2]), max(s[0], s[2])))
        return edges

    # -----------------------------------------------------------------------
    # B. Edge validation
    # -----------------------------------------------------------------------

    def _select_valid_edges(self, edges, points, colors, car_x, car_y, car_yaw):
        """
        Filtra gli edge della triangolazione per ottenere solo quelli validi.

        Strategia in due passate:
        1) CANDIDATI: mantiene solo edge che collegano un cono blu a uno
           giallo, con lunghezza tra track_width_min e track_width_max.
        2) BEST-MATCH: per ogni cono, seleziona il partner più vicino sul
           lato opposto. Gli edge non-best vengono comunque tenuti se la
           loro lunghezza è entro il 130% del best (tolleranza per le curve
           dove un cono può avere due accoppiamenti legittimi).

        Parametri:
            edges: set di (i, j) da Delaunay.
            points: coordinate Nx2 dei coni.
            colors: array con il colore di ogni cono (BLUE=0, YELLOW=1).
            car_x, car_y, car_yaw: posa veicolo (non usata qui, riservata).

        Ritorna:
            (valid_edges, candidate_edges): due liste di tuple (i, j).
        """
        w_min = self._p('track_width_min')
        w_max = self._p('track_width_max')

        # First pass: keep blue-yellow edges within distance bounds
        candidate = []
        for i, j in edges:
            if colors[i] == colors[j]:
                continue  # same-color edge
            dist = np.linalg.norm(points[i] - points[j])
            if dist < w_min or dist > w_max:
                continue
            candidate.append((i, j, dist))

        # Second pass: for each cone keep only the best (shortest) match
        # on the opposite side to reduce spurious long cross-connections.
        best_for = {}  # cone_index -> (partner, dist, edge_tuple)
        for i, j, d in candidate:
            for a, b in [(i, j), (j, i)]:
                prev = best_for.get(a)
                if prev is None or d < prev[1]:
                    best_for[a] = (b, d, (min(i, j), max(i, j)))

        valid_set = set()
        for _, (_, _, edge) in best_for.items():
            valid_set.add(edge)

        # Also keep edges that are close to the best (within 30%) to not
        # lose valid connections at curves where two matches are reasonable.
        valid = []
        for i, j, d in candidate:
            key = (min(i, j), max(i, j))
            if key in valid_set:
                valid.append((i, j))
                continue
            best_i = best_for.get(i)
            best_j = best_for.get(j)
            thr_i = best_i[1] * 1.3 if best_i else 0
            thr_j = best_j[1] * 1.3 if best_j else 0
            if d <= thr_i or d <= thr_j:
                valid.append((i, j))

        return valid, [(i, j) for i, j, _ in candidate]

    # -----------------------------------------------------------------------
    # C. Midpoints
    # -----------------------------------------------------------------------

    def _compute_midpoints(self, edges, points: np.ndarray) -> np.ndarray:
        """
        Calcola i midpoint (punti medi) degli edge validi e li deduplica.

        Ogni edge valido collega un cono blu a uno giallo: il suo punto
        medio rappresenta un'approssimazione del centro pista in quel
        punto. Quando più edge convergono nella stessa zona, si generano
        midpoint quasi sovrapposti. Il merge greedy raggruppa quelli
        entro midpoint_merge_distance e li media, producendo un singolo
        punto rappresentativo per cluster.

        Parametri:
            edges: lista di (i, j) degli edge validati.
            points: coordinate Nx2 dei coni.

        Ritorna:
            np.ndarray Mx2: midpoint deduplicati.
        """
        if not edges:
            return np.empty((0, 2))

        raw = np.array([
            (points[i] + points[j]) / 2.0 for i, j in edges
        ])

        merge_dist = self._p('midpoint_merge_distance')
        if merge_dist <= 0 or len(raw) < 2:
            return raw

        # Greedy merge: iterate and average close points
        merged = []
        used = np.zeros(len(raw), dtype=bool)
        for k in range(len(raw)):
            if used[k]:
                continue
            cluster = [raw[k]]
            used[k] = True
            for m in range(k + 1, len(raw)):
                if used[m]:
                    continue
                if np.linalg.norm(raw[k] - raw[m]) < merge_dist:
                    cluster.append(raw[m])
                    used[m] = True
            merged.append(np.mean(cluster, axis=0))
        return np.array(merged)

    # -----------------------------------------------------------------------
    # D. Ordering
    # -----------------------------------------------------------------------

    def _order_midpoints(self, midpoints: np.ndarray,
                         car_x, car_y, car_yaw) -> np.ndarray:
        """
        Ordina i midpoint in una sequenza coerente con la direzione della pista.

        Algoritmo greedy direzionale migliorato rispetto al nearest-neighbor puro:
        1. Parte dal midpoint più vicino all'auto.
        2. Ad ogni passo, valuta tutti i midpoint non visitati entro
           max_progression_step metri e assegna uno score combinato:
           score = distanza + ordering_direction_weight * deviazione_angolare * distanza
        3. Sceglie il candidato con score minimo (vicino E nella direzione giusta).
        4. Aggiorna l'heading corrente in base all'ultimo segmento percorso.

        Dopo l'ordinamento, tenta di chiudere il loop se l'ultimo punto
        è abbastanza vicino al primo (piste chiuse in Formula Student).

        Parametri:
            midpoints: array Mx2 dei midpoint deduplicati.
            car_x, car_y: posizione del veicolo.
            car_yaw: heading del veicolo (usato come heading iniziale).

        Ritorna:
            np.ndarray: midpoint ordinati lungo la pista.
        """
        max_step = self._p('max_progression_step')
        dir_w = self._p('ordering_direction_weight')
        n = len(midpoints)
        if n == 0:
            return midpoints

        # Start from midpoint closest to the car
        dists_to_car = np.hypot(
            midpoints[:, 0] - car_x, midpoints[:, 1] - car_y)
        cur = int(np.argmin(dists_to_car))

        ordered_idx = [cur]
        visited = {cur}
        heading = car_yaw  # initial heading estimate

        while len(ordered_idx) < n:
            cp = midpoints[ordered_idx[-1]]
            best_idx = -1
            best_score = float('inf')

            for k in range(n):
                if k in visited:
                    continue
                diff = midpoints[k] - cp
                d = np.linalg.norm(diff)
                if d > max_step or d < 1e-6:
                    continue
                # Angular penalty
                angle_to_k = math.atan2(diff[1], diff[0])
                ang_dev = abs(angle_diff(angle_to_k, heading))
                # Score: lower is better.  distance + weighted angular dev
                score = d + dir_w * ang_dev * d
                if score < best_score:
                    best_score = score
                    best_idx = k

            if best_idx < 0:
                break

            # Update heading from last segment
            diff = midpoints[best_idx] - cp
            heading = math.atan2(diff[1], diff[0])
            ordered_idx.append(best_idx)
            visited.add(best_idx)

        result = midpoints[ordered_idx]

        # Attempt to close the loop
        if len(result) >= 6:
            gap = np.linalg.norm(result[-1] - result[0])
            if gap < max_step:
                result = np.vstack([result, result[0:1]])

        return result

    # -----------------------------------------------------------------------
    # E. Smoothing
    # -----------------------------------------------------------------------

    def _smooth_centerline(self, pts: np.ndarray) -> np.ndarray:
        """
        Applica smoothing con spline cubica e ricampiona a passo costante.

        Usa scipy splprep/splev per fittare una curva parametrica cubica
        (k=3) sui midpoint ordinati. Il parametro di smoothing s=N*0.5
        bilancia fedeltà ai dati e regolarità. Poi ricampiona la spline
        a intervalli di smoothing_resolution metri per produrre una
        traiettoria regolare e adatta a planner/controller.

        Se il smoothing è disabilitato, ci sono meno di 4 punti, o la
        spline fallisce, ritorna la polilinea originale come fallback.

        Parametri:
            pts: array Nx2 dei midpoint ordinati.

        Ritorna:
            np.ndarray: centerline liscia e ricampionata.
        """
        if not self._p('smoothing_enabled') or len(pts) < 4:
            return pts

        res = self._p('smoothing_resolution')
        try:
            # Parametric spline fit
            tck, u = splprep([pts[:, 0], pts[:, 1]], s=len(pts)* 0.5, k=3)
            # Estimate total arc length
            diffs = np.diff(pts, axis=0)
            arc = np.sum(np.hypot(diffs[:, 0], diffs[:, 1]))
            n_samples = max(int(arc / res), len(pts))
            u_new = np.linspace(0, 1, n_samples)
            sx, sy = splev(u_new, tck)
            return np.column_stack([sx, sy])
        except Exception as e:
            self.get_logger().debug(f"Spline failed, using polyline: {e}")
            return pts

    # -----------------------------------------------------------------------
    # F. Temporal fusion
    # -----------------------------------------------------------------------

    def _temporal_fuse(self, new_cl: np.ndarray,
                       car_x: float, car_y: float) -> np.ndarray:
        """
        Fonde la centerline appena calcolata con quella del frame precedente.

        Per ridurre il jitter (oscillazioni tra frame consecutivi), ogni
        punto della nuova centerline viene mescolato con il punto più
        vicino della centerline precedente tramite media pesata:
            punto_fuso = alpha * nuovo + (1-alpha) * precedente

        L'alpha è adattivo: più alto vicino all'auto (reattivo, segue i
        dati nuovi) e più basso lontano (stabile, resiste al rumore).
        Formula: local_alpha = min(1.0, alpha + 0.3 * exp(-dist_car / 10))

        Il blend avviene solo se il punto precedente più vicino è entro
        3.0m (altrimenti la topologia è cambiata troppo e non ha senso
        interpolare).

        Parametri:
            new_cl: centerline nuova (array Nx2).
            car_x, car_y: posizione del veicolo per calcolare l'alpha adattivo.

        Ritorna:
            np.ndarray: centerline fusa e stabilizzata.
        """
        alpha = self._p('temporal_filter_alpha')
        prev = self._prev_centerline

        if prev is None or len(prev) < 2 or alpha >= 1.0:
            return new_cl
        if alpha <= 0.0:
            return prev

        # For each new point find nearest in previous centerline
        fused = new_cl.copy()
        for k in range(len(fused)):
            dists = np.linalg.norm(prev - fused[k], axis=1)
            j = int(np.argmin(dists))
            if dists[j] < 3.0:  # only blend if close enough
                # Higher alpha near the car (responsive), lower far away
                d_car = math.hypot(fused[k, 0] - car_x,
                                   fused[k, 1] - car_y)
                local_alpha = min(1.0, alpha + 0.3 * math.exp(-d_car / 10.0))
                fused[k] = local_alpha * fused[k] + (1 - local_alpha) * prev[j]
        return fused

    # -----------------------------------------------------------------------
    # Publishing
    # -----------------------------------------------------------------------

    def _publish_path(self, pts: np.ndarray):
        """
        Pubblica la centerline finale come messaggio nav_msgs/Path.

        Crea un PoseStamped per ogni punto della centerline, con z=0
        e orientamento neutro (quaternione identità). Il frame_id viene
        impostato al frame corrente dei coni (base_footprint o map).

        Parametri:
            pts: array Nx2 con i punti della centerline finale.
        """
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._cones_frame
        for x, y in pts:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        self._pub_path.publish(msg)

    def _publish_cone_markers(self, cones):
        """
        Pubblica i coni come cilindri colorati in RViz (MarkerArray).

        Ogni cono viene visualizzato come un cilindro 3D: blu per i coni
        sinistri, giallo per i coni destri. Dimensioni: diametro 0.2m,
        altezza 0.3m, centrato a z=0.15m dal suolo.

        Parametri:
            cones: lista di tuple (x, y, colore) dei coni filtrati.
        """
        ma = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        for i, (cx, cy, color) in enumerate(cones):
            m = Marker()
            m.header.frame_id = self._cones_frame
            m.header.stamp = stamp
            m.ns = 'cones'
            m.id = i
            m.type = Marker.CYLINDER
            m.action = Marker.ADD
            m.pose.position.x = float(cx)
            m.pose.position.y = float(cy)
            m.pose.position.z = 0.15
            m.scale.x = m.scale.y = 0.2
            m.scale.z = 0.3
            m.color = rgba(0, 0, 1) if color == self.BLUE else rgba(1, 1, 0)
            ma.markers.append(m)
        self._pub_cones.publish(ma)

    # -----------------------------------------------------------------------
    # H. Debug markers
    # -----------------------------------------------------------------------

    def _publish_debug_markers(self, points, candidate_edges, valid_edges,
                                raw_mids, ordered_mids, smooth_cl):
        """
        Pubblica marker di debug per visualizzare ogni fase della pipeline.

        Tutti i marker vengono inviati su un unico topic (/debug/centerline)
        con namespace separati, così in RViz si può attivare/disattivare
        ogni layer indipendentemente. Prima di ogni pubblicazione si inviano
        marker DELETEALL per evitare marker residui dai frame precedenti.

        Layer pubblicati:
        - edges_candidate (grigio): tutti gli edge blu↔giallo entro le soglie
        - edges_valid (verde): edge sopravvissuti al filtro best-match
        - mid_raw (arancione): midpoint grezzi dopo merge
        - mid_ordered (ciano): midpoint nell'ordine della centerline
        - cl_smooth (magenta): centerline liscia finale

        Parametri:
            points: coordinate Nx2 dei coni.
            candidate_edges: lista (i,j) degli edge candidati.
            valid_edges: lista (i,j) degli edge validati.
            raw_mids: array Mx2 dei midpoint deduplicati.
            ordered_mids: array dei midpoint ordinati.
            smooth_cl: array della centerline smoothata.
        """
        ma = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        frame = self._cones_frame

        # We can use a single DELETEALL marker at the beginning
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        ma.markers.append(delete_marker)

        # Candidate edges (gray, thin, LINE_LIST)
        if candidate_edges:
            m = Marker()
            m.header.frame_id = frame
            m.header.stamp = stamp
            m.ns = 'edges_candidate'
            m.id = 1
            m.type = Marker.LINE_LIST
            m.action = Marker.ADD
            m.scale.x = 0.03
            m.color = rgba(0.6, 0.6, 0.6, 0.4)
            for i, j in candidate_edges:
                p1 = PoseStamped().pose.position
                p1.x, p1.y, p1.z = float(points[i][0]), float(points[i][1]), 0.05
                p2 = PoseStamped().pose.position
                p2.x, p2.y, p2.z = float(points[j][0]), float(points[j][1]), 0.05
                m.points.append(p1)
                m.points.append(p2)
            ma.markers.append(m)

        # Valid edges (green, LINE_LIST)
        if valid_edges:
            m = Marker()
            m.header.frame_id = frame
            m.header.stamp = stamp
            m.ns = 'edges_valid'
            m.id = 2
            m.type = Marker.LINE_LIST
            m.action = Marker.ADD
            m.scale.x = 0.06
            m.color = rgba(0, 1, 0, 0.8)
            for i, j in valid_edges:
                p1 = PoseStamped().pose.position
                p1.x, p1.y, p1.z = float(points[i][0]), float(points[i][1]), 0.05
                p2 = PoseStamped().pose.position
                p2.x, p2.y, p2.z = float(points[j][0]), float(points[j][1]), 0.05
                m.points.append(p1)
                m.points.append(p2)
            ma.markers.append(m)

        # Raw midpoints (orange, SPHERE_LIST)
        if len(raw_mids) > 0:
            m = Marker()
            m.header.frame_id = frame
            m.header.stamp = stamp
            m.ns = 'mid_raw'
            m.id = 3
            m.type = Marker.SPHERE_LIST
            m.action = Marker.ADD
            m.scale.x = 0.2
            m.scale.y = 0.2
            m.scale.z = 0.2
            m.color = rgba(1.0, 0.5, 0.0, 0.9)
            for pt in raw_mids:
                p = PoseStamped().pose.position
                p.x, p.y, p.z = float(pt[0]), float(pt[1]), 0.15
                m.points.append(p)
            ma.markers.append(m)

        # Ordered midpoints (cyan, SPHERE_LIST)
        if len(ordered_mids) > 0:
            m = Marker()
            m.header.frame_id = frame
            m.header.stamp = stamp
            m.ns = 'mid_ordered'
            m.id = 4
            m.type = Marker.SPHERE_LIST
            m.action = Marker.ADD
            m.scale.x = 0.25
            m.scale.y = 0.25
            m.scale.z = 0.25
            m.color = rgba(0, 1, 1, 0.9)
            for pt in ordered_mids:
                p = PoseStamped().pose.position
                p.x, p.y, p.z = float(pt[0]), float(pt[1]), 0.2
                m.points.append(p)
            ma.markers.append(m)

        # Smooth centerline (magenta, LINE_STRIP)
        if len(smooth_cl) >= 2:
            m = Marker()
            m.header.frame_id = frame
            m.header.stamp = stamp
            m.ns = 'cl_smooth'
            m.id = 5
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.scale.x = 0.08
            m.color = rgba(1, 0, 1, 0.9)
            for pt in smooth_cl:
                p = PoseStamped().pose.position
                p.x, p.y, p.z = float(pt[0]), float(pt[1]), 0.1
                m.points.append(p)
            ma.markers.append(m)

        self._pub_debug.publish(ma)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = CenterlineNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
