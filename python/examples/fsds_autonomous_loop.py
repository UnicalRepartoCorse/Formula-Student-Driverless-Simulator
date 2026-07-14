"""
=============================================================================
 FSDS Autonomous Loop - Master Script di Integrazione
 Formula Student Driverless Simulator
=============================================================================

 Questo script implementa il ciclo chiuso completo di guida autonoma:

   [Simulatore] --> [Percezione] --> [Pianificazione] --> [Controllo] --> [Simulatore]
        ^                                                                      |
        |______________________________________________________________________|

 Pipeline per frame:
   1. simGetImages()          → acquisisce i frame stereo dal simulatore
   2. mock_yolo_detect()      → (PLACEHOLDER) rileva i coni nelle immagini
   3. ConeTracker.update()    → stabilizza i rilevamenti (anti falsi negativi)
   4. PathPlanner             → calcola la centerline (midpoint matching)
   5. PurePursuitController   → calcola il comando sterzo normalizzato
   6. setCarControls()        → invia throttle e sterzo al simulatore

 Prerequisiti:
   - Simulatore FSDS in esecuzione
   - settings.json con cam_left e cam_right configurate (vedi sotto)
   - pip install msgpack-rpc-python numpy opencv-python scipy

 Avvio:
   python fsds_autonomous_loop.py

 Interruzione pulita:
   Premi Ctrl+C → la vettura si ferma e le API vengono rilasciate.

 settings.json minimo richiesto:
   {
     "SettingsVersion": 1.2,
     "SimMode": "Car",
     "Vehicles": {
       "FSCar": {
         "VehicleType": "Car",
         "AutoCreate": true,
         "Cameras": {
           "cam_left":  {"X": 1.5, "Y": -0.1, "Z": -0.8},
           "cam_right": {"X": 1.5, "Y":  0.1, "Z": -0.8}
         }
       }
     }
   }
=============================================================================
"""

import sys
import os
import time
import numpy as np

# Aggiunge la directory padre al path per trovare il package fsds e i moduli
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Client nativo FSDS
import fsds

# Moduli della pipeline (sviluppati nei file separati)
from perception.cone_tracker import ConeTracker
from perception.path_planner  import PathPlanner
from perception.pure_pursuit  import PurePursuitController


# =============================================================================
# CONFIGURAZIONE GLOBALE
# =============================================================================

# Parametri telecamere
CAM_LEFT  = 'cam_left'
CAM_RIGHT = 'cam_right'

# Parametri tracker
TRACKER_DISTANCE_THRESHOLD = 1.5   # metri: soglia per abbinare un cono
TRACKER_MAX_FRAMES_LOST    = 5     # frame: pazienza prima di rimuovere un cono

# Parametri controller
CONTROLLER_WHEELBASE         = 1.53  # metri: passo della vettura
CONTROLLER_LOOKAHEAD         = 4.0   # metri: distanza di mira Pure Pursuit
CONTROLLER_MAX_STEERING_RAD  = 0.52  # radianti: angolo massimo ruote (~30°)

# Parametri di guida
THROTTLE_CONSTANT = 0.20  # valore fisso [0.0, 1.0] per il test a velocità costante
LOOP_SLEEP_SEC    = 0.03  # secondi: pausa tra un ciclo e l'altro (~33 Hz)

# ID colore coni (devono corrispondere all'output del modello YOLO)
COLOR_BLUE   = 0  # coni blu   → delimitazione SINISTRA
COLOR_YELLOW = 1  # coni gialli → delimitazione DESTRA


# =============================================================================
# FUNZIONE DI DECODIFICA IMMAGINE (da AirSim a OpenCV BGR)
# =============================================================================

def decode_image(response: fsds.ImageResponse) -> np.ndarray:
    """
    Converte un ImageResponse FSDS in un array NumPy BGR.

    ATTENZIONE: AirSim con compress=True restituisce PNG compresso.
    Importiamo cv2 qui per evitare crash se non installato (non è critico
    per la pipeline di controllo, serve solo per il debug visivo).
    """
    import cv2
    img_1d  = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
    img_bgr = cv2.imdecode(img_1d, cv2.IMREAD_COLOR)
    return img_bgr


# =============================================================================
# MOCK YOLO — PLACEHOLDER PER IL RILEVAMENTO CONI
# =============================================================================

def mock_yolo_detect(img_left: np.ndarray, img_right: np.ndarray) -> list:
    """
    PLACEHOLDER: Qui andrà il modello di rilevamento reale (es. YOLOv8).

    Questa funzione simula l'output di YOLO restituendo una lista fissa
    di coni in coordinate metriche (X, Z) relative alla vettura.
    Le coordinate sono già triangolate (come farebbe un vero modello stereo
    o un algoritmo di stima della profondità a partire dalla coppia stereo).

    Formato output:
        Lista di tuple (x, z, color_id) dove:
          x        = spostamento laterale in metri (+ destra, - sinistra)
          z        = distanza frontale  in metri (avanzamento)
          color_id = 0 (blu/sinistra), 1 (giallo/destra)

    TODO: Sostituire il corpo di questa funzione con:
        results = yolo_model.predict(img_left)       # inferenza sul frame sinistro
        detections = stereo_triangulate(results, img_left, img_right)
        return [(x, z, class_id) for x, z, class_id in detections]

    Args:
        img_left:  Frame BGR della camera sinistra (NumPy array).
        img_right: Frame BGR della camera destra  (NumPy array).

    Returns:
        Lista di tuple (x, z, color_id) che simulano un piccolo tracciato.
    """
    # Coni finti: 4 blu a sinistra, 4 gialli a destra, che curvano a destra
    return [
        # (  x,     z, color_id)
        (-1.6,  3.0, COLOR_BLUE),
        (-1.4,  6.0, COLOR_BLUE),
        (-1.0,  9.0, COLOR_BLUE),
        (-0.5, 12.0, COLOR_BLUE),
        ( 1.9,  3.0, COLOR_YELLOW),
        ( 2.1,  6.0, COLOR_YELLOW),
        ( 2.5,  9.0, COLOR_YELLOW),
        ( 3.0, 12.0, COLOR_YELLOW),
    ]


# =============================================================================
# FUNZIONE HELPER: ACQUISIZIONE FRAME STEREO SINCRONIZZATO
# =============================================================================

def get_stereo_frames(client: fsds.FSDSClient):
    """
    Acquisisce i frame delle due camere in una singola chiamata RPC.

    L'uso di una sola chiamata simGetImages con due ImageRequest garantisce
    la sincronizzazione temporale tra i due frame (stesso tick di simulazione).

    Returns:
        Tuple (img_left, img_right) come array NumPy BGR.
        Restituisce (None, None) se i frame sono vuoti o non validi.
    """
    responses = client.simGetImages(
        [
            fsds.ImageRequest(
                camera_name=CAM_LEFT,
                image_type=fsds.ImageType.Scene,
                pixels_as_float=False,
                compress=True
            ),
            fsds.ImageRequest(
                camera_name=CAM_RIGHT,
                image_type=fsds.ImageType.Scene,
                pixels_as_float=False,
                compress=True
            ),
        ],
        vehicle_name='FSCar'
    )

    # Verifica completezza della risposta
    if len(responses) < 2:
        return None, None

    resp_l, resp_r = responses[0], responses[1]

    # Verifica che entrambi i frame siano non vuoti
    if len(resp_l.image_data_uint8) == 0 or len(resp_r.image_data_uint8) == 0:
        return None, None

    return decode_image(resp_l), decode_image(resp_r)


# =============================================================================
# FUNZIONE DI ARRESTO SICURO
# =============================================================================

def stop_vehicle(client: fsds.FSDSClient):
    """
    Invia un comando di fermo completo alla vettura.
    Chiamato in caso di interruzione o errore critico.
    """
    stop_controls = fsds.CarControls()
    stop_controls.throttle = 0.0
    stop_controls.steering  = 0.0
    stop_controls.brake     = 1.0  # freno massimo per sicurezza
    client.setCarControls(stop_controls)
    print("[SAFE STOP] Vettura fermata.")


# =============================================================================
# MAIN - CICLO AUTONOMO CHIUSO
# =============================================================================

def main():
    print("=" * 65)
    print("  FSDS Autonomous Loop — Avvio Pipeline Completa")
    print("=" * 65)

    # ------------------------------------------------------------------
    # SETUP: Connessione al simulatore
    # ------------------------------------------------------------------
    print("\n[SETUP] Connessione al simulatore FSDS...")
    client = fsds.FSDSClient()
    client.confirmConnection()
    client.enableApiControl(True)
    print("[SETUP] Connesso! API Control abilitato.")

    # ------------------------------------------------------------------
    # INIZIALIZZAZIONE: Moduli della pipeline
    # ------------------------------------------------------------------
    tracker    = ConeTracker(
        distance_threshold=TRACKER_DISTANCE_THRESHOLD,
        max_frames_lost=TRACKER_MAX_FRAMES_LOST
    )
    planner    = PathPlanner()
    controller = PurePursuitController(
        wheelbase=CONTROLLER_WHEELBASE,
        lookahead_distance=CONTROLLER_LOOKAHEAD,
        max_steering_angle=CONTROLLER_MAX_STEERING_RAD
    )

    print(f"[SETUP] Tracker:    {tracker}")
    print(f"[SETUP] Planner:    {planner.__class__.__name__}")
    print(f"[SETUP] Controller: {controller}")
    print("\n[INFO] Ciclo autonomo avviato. Premi Ctrl+C per fermare.\n")

    frame_count  = 0
    errori_frame = 0

    # ------------------------------------------------------------------
    # CICLO PRINCIPALE CHIUSO: Perception → Planning → Control → Actuation
    # ------------------------------------------------------------------
    try:
        while True:
            frame_count += 1
            t_inizio = time.time()

            # ==============================================================
            # STEP 1 — ACQUISIZIONE FRAME STEREO
            # ==============================================================
            img_left, img_right = get_stereo_frames(client)

            if img_left is None or img_right is None:
                # Frame non disponibile: saltiamo questo ciclo
                errori_frame += 1
                if errori_frame % 20 == 0:
                    print(f"[WARN] Frame #{frame_count}: dati camera non disponibili "
                          f"({errori_frame} frame saltati totali).")
                time.sleep(LOOP_SLEEP_SEC)
                continue

            # ==============================================================
            # STEP 2 — RILEVAMENTO CONI (MOCK YOLO)
            # ==============================================================
            # TODO: Qui andrà il modello YOLO reale + triangolazione stereo.
            # Per ora mock_yolo_detect() restituisce coordinate finte.
            yolo_output = mock_yolo_detect(img_left, img_right)
            # yolo_output: lista di (x, z, color_id) in coordinate metriche

            # ==============================================================
            # STEP 3 — TRACKING (stabilizzazione rilevamenti)
            # ==============================================================
            active_cones = tracker.update(yolo_output)
            # active_cones: lista di TrackedCone con x, y (=z), color_id

            # ==============================================================
            # STEP 4 — SEPARAZIONE PER COLORE
            # ==============================================================
            # Nota: in ConeTracker, l'asse "y" del cono corrisponde all'asse Z
            # del sistema di riferimento FSDS (profondità/avanzamento).
            blue_cones   = [(c.x, c.y) for c in active_cones if c.color_id == COLOR_BLUE]
            yellow_cones = [(c.x, c.y) for c in active_cones if c.color_id == COLOR_YELLOW]

            # ==============================================================
            # STEP 5 — PIANIFICAZIONE CENTERLINE
            # ==============================================================
            centerline = planner.compute_centerline(blue_cones, yellow_cones)

            # ==============================================================
            # STEP 6 — CALCOLO STERZO (Pure Pursuit)
            # ==============================================================
            if centerline:
                steering_cmd = controller.calculate_steering(centerline)
            else:
                # Nessuna centerline disponibile: va dritto
                steering_cmd = 0.0
                if frame_count % 30 == 0:
                    print(f"[WARN] Frame #{frame_count}: centerline vuota, sterzo = 0.")

            # ==============================================================
            # STEP 7 — ATTUAZIONE: Invio comandi al simulatore
            # ==============================================================
            controls          = fsds.CarControls()
            controls.throttle = THROTTLE_CONSTANT  # velocità costante per il test
            controls.steering  = steering_cmd       # sterzo calcolato da Pure Pursuit
            controls.brake    = 0.0

            client.setCarControls(controls)

            # ==============================================================
            # TELEMETRIA (ogni 30 frame, ~1 secondo a 33Hz)
            # ==============================================================
            if frame_count % 30 == 0:
                t_ciclo_ms = (time.time() - t_inizio) * 1000
                n_blu    = len(blue_cones)
                n_gialli = len(yellow_cones)
                n_wp     = len(centerline)
                print(
                    f"[Frame {frame_count:>5}] "
                    f"Blu={n_blu} | Gialli={n_gialli} | "
                    f"Waypoint={n_wp} | "
                    f"Sterzo={steering_cmd:+.3f} | "
                    f"Throttle={THROTTLE_CONSTANT:.2f} | "
                    f"Ciclo={t_ciclo_ms:.1f}ms"
                )

            # Pausa per non saturare la CPU e rispettare il rate del simulatore
            time.sleep(LOOP_SLEEP_SEC)

    # ------------------------------------------------------------------
    # INTERRUZIONE PULITA: Ctrl+C
    # ------------------------------------------------------------------
    except KeyboardInterrupt:
        print("\n\n[INFO] Interruzione da tastiera (Ctrl+C) rilevata.")
        print(f"[INFO] Frame processati: {frame_count}")

    # ------------------------------------------------------------------
    # GESTIONE ERRORI IMPREVISTI
    # ------------------------------------------------------------------
    except Exception as e:
        print(f"\n[ERRORE CRITICO] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

    # ------------------------------------------------------------------
    # CLEANUP: Eseguito sempre (anche in caso di errore)
    # ------------------------------------------------------------------
    finally:
        print("\n[CLEANUP] Rilascio controllo vettura...")
        stop_vehicle(client)
        client.enableApiControl(False)
        print("[CLEANUP] API Control rilasciato. Script terminato.")
        print("=" * 65)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    main()
