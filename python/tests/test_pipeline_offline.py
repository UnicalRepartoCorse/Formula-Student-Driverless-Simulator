"""
=============================================================================
 Test Offline della Pipeline Autonoma - Senza Simulatore FSDS
=============================================================================

 Scopo:
   Verificare l'intera catena logica:
     MockClient → mock_yolo → ConeTracker → PathPlanner
                → PurePursuitController → CarControls

   Questo test NON richiede il simulatore FSDS in esecuzione.
   Il client reale viene sostituito da un MockClient che simula
   le risposte del simulatore con dati sintetici.

 Come eseguire:
   python python/tests/test_pipeline_offline.py

 Scenari testati:
   1. Rettilineo  → sterzo vicino a 0
   2. Curva destra → sterzo positivo
   3. Curva sinistra → sterzo negativo
   4. Falso negativo (cono perso per 1 frame) → tracker lo mantiene
   5. N frame in ciclo chiuso → nessun crash, output sempre in [-1, 1]
=============================================================================
"""

import sys
import os
import time
import numpy as np

# Forza stdout in UTF-8 per evitare UnicodeEncodeError su Windows (cp1252)
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Aggiunge la directory python/ al path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from perception.cone_tracker import ConeTracker
from perception.path_planner  import PathPlanner
from perception.pure_pursuit  import PurePursuitController


# =============================================================================
# MOCK: Sostituti del client FSDS e dei tipi AirSim
# =============================================================================

class MockImageResponse:
    """
    Simula un fsds.ImageResponse con dati PNG minimi validi.
    Usa un PNG 1x1 pixel nero hardcoded come bytes literal per evitare
    qualsiasi dipendenza da cv2 o PIL in fase di test offline.
    """

    # PNG 1x1 pixel nero (RGB=0,0,0) hardcoded - nessuna dipendenza esterna
    _PNG_1X1_BLACK = (
        b'\x89PNG\r\n\x1a\n'
        b'\x00\x00\x00\rIHDR'
        b'\x00\x00\x00\x01\x00\x00\x00\x01'
        b'\x08\x02\x00\x00\x00'
        b'\x90wS\xde'
        b'\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N'
        b'\x00\x00\x00\x00IEND\xaeB`\x82'
    )

    def __init__(self):
        self.image_data_uint8 = self._PNG_1X1_BLACK
        self.camera_name = 'mock_cam'



class MockCarControls:
    """Simula fsds.CarControls."""
    def __init__(self):
        self.throttle = 0.0
        self.steering  = 0.0
        self.brake    = 0.0

    def __repr__(self):
        return (f"CarControls(throttle={self.throttle:.2f}, "
                f"steering={self.steering:+.3f}, "
                f"brake={self.brake:.2f})")


class MockFSDSClient:
    """
    Sostituisce fsds.FSDSClient senza connessione di rete.

    Registra tutti i comandi inviati per poterli verificare nel test.
    """

    def __init__(self):
        self.controls_history = []   # storico di tutti i CarControls inviati
        self.api_enabled      = False
        self._call_count      = 0

    def confirmConnection(self):
        print("  [MOCK] Ping al simulatore → OK (connessione simulata)")

    def enableApiControl(self, enabled, vehicle_name='FSCar'):
        self.api_enabled = enabled
        stato = "ABILITATO" if enabled else "DISABILITATO"
        print(f"  [MOCK] API Control {stato} per '{vehicle_name}'")

    def simGetImages(self, requests, vehicle_name='FSCar'):
        """Restituisce N immagini PNG nere (una per ogni richiesta)."""
        self._call_count += 1
        return [MockImageResponse() for _ in requests]

    def setCarControls(self, controls, vehicle_name='FSCar'):
        """Registra il comando ricevuto nello storico."""
        self.controls_history.append({
            'throttle': controls.throttle,
            'steering':  controls.steering,
            'brake':    controls.brake,
        })


# =============================================================================
# SCENARI DI CONI SINTETICI
# =============================================================================

# Ogni scenario è una lista di tuple (x, z, color_id)
COLOR_BLUE   = 0
COLOR_YELLOW = 1

SCENARIO_RETTILINEO = [
    (-1.75,  4.0, COLOR_BLUE),   (-1.75,  8.0, COLOR_BLUE),
    (-1.75, 12.0, COLOR_BLUE),   (-1.75, 16.0, COLOR_BLUE),
    ( 1.75,  4.0, COLOR_YELLOW), ( 1.75,  8.0, COLOR_YELLOW),
    ( 1.75, 12.0, COLOR_YELLOW), ( 1.75, 16.0, COLOR_YELLOW),
]

SCENARIO_CURVA_DESTRA = [
    (-1.5,  4.0, COLOR_BLUE),   (-0.8,  8.0, COLOR_BLUE),
    ( 0.2, 12.0, COLOR_BLUE),   ( 1.5, 15.0, COLOR_BLUE),
    ( 2.0,  4.0, COLOR_YELLOW), ( 2.8,  8.0, COLOR_YELLOW),
    ( 3.8, 12.0, COLOR_YELLOW), ( 5.0, 15.0, COLOR_YELLOW),
]

SCENARIO_CURVA_SINISTRA = [
    (-5.0, 15.0, COLOR_BLUE),   (-3.8, 12.0, COLOR_BLUE),
    (-2.8,  8.0, COLOR_BLUE),   (-2.0,  4.0, COLOR_BLUE),
    (-1.5, 15.0, COLOR_YELLOW), (-0.2, 12.0, COLOR_YELLOW),
    ( 0.8,  8.0, COLOR_YELLOW), ( 1.5,  4.0, COLOR_YELLOW),
]


# =============================================================================
# FUNZIONE HELPER: Esegue un singolo ciclo della pipeline
# =============================================================================

def esegui_ciclo(client, tracker, planner, controller, coni_rilevati):
    """
    Esegue un singolo tick della pipeline completa:
      simGetImages → mock_yolo → tracker → planner → controller → setCarControls

    Returns:
        dict con i valori chiave del ciclo (per le asserzioni del test).
    """
    # Step 1 - Acquisizione immagini (mock)
    responses = client.simGetImages(
        [{'camera_name': 'cam_left'}, {'camera_name': 'cam_right'}],
        vehicle_name='FSCar'
    )
    assert len(responses) == 2, "simGetImages deve restituire 2 risposte"

    # Step 2 - YOLO (iniettato dall'esterno nel test)
    yolo_output = coni_rilevati  # in produzione: mock_yolo_detect(img_l, img_r)

    # Step 3 - Tracking
    active_cones = tracker.update(yolo_output)

    # Step 4 - Separazione colori
    blue_cones   = [(c.x, c.y) for c in active_cones if c.color_id == COLOR_BLUE]
    yellow_cones = [(c.x, c.y) for c in active_cones if c.color_id == COLOR_YELLOW]

    # Step 5 - Pianificazione
    centerline = planner.compute_centerline(blue_cones, yellow_cones)

    # Step 6 - Controllo
    steering_cmd = controller.calculate_steering(centerline) if centerline else 0.0

    # Step 7 - Attuazione
    controls          = MockCarControls()
    controls.throttle = 0.20
    controls.steering  = steering_cmd
    controls.brake    = 0.0
    client.setCarControls(controls)

    return {
        'n_coni_attivi': len(active_cones),
        'n_blu':         len(blue_cones),
        'n_gialli':      len(yellow_cones),
        'n_waypoint':    len(centerline),
        'steering':      steering_cmd,
        'throttle':      controls.throttle,
    }


# =============================================================================
# SUITE DI TEST
# =============================================================================

def test_rettilineo():
    print("\n" + "-" * 55)
    print("  SCENARIO 1: Rettilineo")
    print("-" * 55)

    client     = MockFSDSClient()
    tracker    = ConeTracker(distance_threshold=1.5, max_frames_lost=5)
    planner    = PathPlanner()
    controller = PurePursuitController(wheelbase=1.53, lookahead_distance=4.0)

    risultato = esegui_ciclo(client, tracker, planner, controller, SCENARIO_RETTILINEO)

    print(f"  Coni attivi : {risultato['n_coni_attivi']} (Blu={risultato['n_blu']}, Gialli={risultato['n_gialli']})")
    print(f"  Waypoint    : {risultato['n_waypoint']}")
    print(f"  Sterzo      : {risultato['steering']:+.4f}  (atteso: ~0.0)")
    print(f"  Throttle    : {risultato['throttle']:.2f}")

    assert risultato['n_waypoint'] > 0,          "Centerline vuota su rettilineo!"
    assert abs(risultato['steering']) < 0.05,    "Sterzo non nullo su rettilineo!"
    assert -1.0 <= risultato['steering'] <= 1.0, "Sterzo fuori range [-1, 1]!"
    print("  [OK] Rettilineo verificato.")


def test_curva_destra():
    print("\n" + "-" * 55)
    print("  SCENARIO 2: Curva Destra")
    print("-" * 55)

    client     = MockFSDSClient()
    tracker    = ConeTracker(distance_threshold=1.5, max_frames_lost=5)
    planner    = PathPlanner()
    controller = PurePursuitController(wheelbase=1.53, lookahead_distance=4.0)

    risultato = esegui_ciclo(client, tracker, planner, controller, SCENARIO_CURVA_DESTRA)

    print(f"  Coni attivi : {risultato['n_coni_attivi']}")
    print(f"  Waypoint    : {risultato['n_waypoint']}")
    print(f"  Sterzo      : {risultato['steering']:+.4f}  (atteso: > 0)")

    assert risultato['steering'] > 0,            "Curva destra deve dare sterzo positivo!"
    assert -1.0 <= risultato['steering'] <= 1.0, "Sterzo fuori range [-1, 1]!"
    print("  [OK] Curva destra verificata.")


def test_curva_sinistra():
    print("\n" + "-" * 55)
    print("  SCENARIO 3: Curva Sinistra")
    print("-" * 55)

    client     = MockFSDSClient()
    tracker    = ConeTracker(distance_threshold=1.5, max_frames_lost=5)
    planner    = PathPlanner()
    controller = PurePursuitController(wheelbase=1.53, lookahead_distance=4.0)

    risultato = esegui_ciclo(client, tracker, planner, controller, SCENARIO_CURVA_SINISTRA)

    print(f"  Coni attivi : {risultato['n_coni_attivi']}")
    print(f"  Waypoint    : {risultato['n_waypoint']}")
    print(f"  Sterzo      : {risultato['steering']:+.4f}  (atteso: < 0)")

    assert risultato['steering'] < 0,            "Curva sinistra deve dare sterzo negativo!"
    assert -1.0 <= risultato['steering'] <= 1.0, "Sterzo fuori range [-1, 1]!"
    print("  [OK] Curva sinistra verificata.")


def test_falso_negativo_tracker():
    print("\n" + "-" * 55)
    print("  SCENARIO 4: Falso Negativo (cono perso per 1 frame)")
    print("-" * 55)

    client     = MockFSDSClient()
    tracker    = ConeTracker(distance_threshold=1.5, max_frames_lost=5)
    planner    = PathPlanner()
    controller = PurePursuitController(wheelbase=1.53, lookahead_distance=4.0)

    # Frame 1: tutti i coni visibili
    r1 = esegui_ciclo(client, tracker, planner, controller, SCENARIO_RETTILINEO)
    print(f"  Frame 1 → Coni attivi: {r1['n_coni_attivi']} | Sterzo: {r1['steering']:+.4f}")

    # Frame 2: un cono blu rimosso (falso negativo YOLO)
    coni_mancante = [c for c in SCENARIO_RETTILINEO if not (c[0] == -1.75 and c[1] == 4.0)]
    r2 = esegui_ciclo(client, tracker, planner, controller, coni_mancante)
    print(f"  Frame 2 → Coni attivi: {r2['n_coni_attivi']} (cono perso ma mantenuto) | Sterzo: {r2['steering']:+.4f}")

    # Frame 3: il cono torna
    r3 = esegui_ciclo(client, tracker, planner, controller, SCENARIO_RETTILINEO)
    print(f"  Frame 3 → Coni attivi: {r3['n_coni_attivi']} (cono recuperato) | Sterzo: {r3['steering']:+.4f}")

    # Il tracker deve aver mantenuto tutti i coni anche al frame 2
    assert r2['n_coni_attivi'] == r1['n_coni_attivi'], \
        "Il tracker avrebbe dovuto mantenere il cono perso!"
    # Al frame 3 tutti i frames_lost devono essere azzerati
    assert r3['n_coni_attivi'] == r1['n_coni_attivi'], "Coni non recuperati al frame 3!"
    print("  [OK] Tracker ha mantenuto il cono durante il falso negativo.")


def test_ciclo_chiuso_n_frame():
    print("\n" + "-" * 55)
    print("  SCENARIO 5: Ciclo chiuso su 50 frame (stress test)")
    print("-" * 55)

    N_FRAME = 50
    client     = MockFSDSClient()
    tracker    = ConeTracker(distance_threshold=1.5, max_frames_lost=5)
    planner    = PathPlanner()
    controller = PurePursuitController(wheelbase=1.53, lookahead_distance=4.0)

    scenari_rotanti = [SCENARIO_RETTILINEO, SCENARIO_CURVA_DESTRA, SCENARIO_CURVA_SINISTRA]
    sterzi = []
    t0 = time.time()

    for i in range(N_FRAME):
        coni = scenari_rotanti[i % len(scenari_rotanti)]
        r = esegui_ciclo(client, tracker, planner, controller, coni)

        # Asserzione critica: lo sterzo non deve mai uscire dal range
        assert -1.0 <= r['steering'] <= 1.0, \
            f"Frame {i}: sterzo {r['steering']} fuori range [-1, 1]!"
        sterzi.append(r['steering'])

    t_totale = (time.time() - t0) * 1000

    t_sicuro = max(t_totale, 0.001)  # evita ZeroDivisionError su macchine molto veloci
    print(f"  Frame processati : {N_FRAME}")
    print(f"  Tempo totale     : {t_totale:.2f} ms  ({t_sicuro/N_FRAME:.3f} ms/frame)")
    print(f"  Frequenza media  : {1000/(t_sicuro/N_FRAME):.0f} Hz")
    print(f"  Comandi inviati  : {len(client.controls_history)}")
    print(f"  Sterzo min/max   : {min(sterzi):+.3f} / {max(sterzi):+.3f}")
    print(f"  Tutti nel range  : [-1.0, 1.0] ✓")
    print(f"  [OK] Stress test su {N_FRAME} frame completato senza errori.")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    print("=" * 55)
    print("  TEST OFFLINE PIPELINE AUTONOMA FSDS")
    print("  (Nessun simulatore richiesto)")
    print("=" * 55)

    test_rettilineo()
    test_curva_destra()
    test_curva_sinistra()
    test_falso_negativo_tracker()
    test_ciclo_chiuso_n_frame()

    print("\n" + "=" * 55)
    print("  TUTTI I TEST SUPERATI CON SUCCESSO")
    print("=" * 55)
