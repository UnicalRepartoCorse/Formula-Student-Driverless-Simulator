"""
=============================================================================
 FSDS Stereo Vision - Acquisizione e visualizzazione in tempo reale
=============================================================================

 Prerequisiti:
   pip install msgpack-rpc-python numpy opencv-python

 settings.json richiesto (Cameras nel veicolo FSCar):
   "cam_left":  { "X": 1.5, "Y": -0.1, "Z": -0.8, "Pitch":0,"Roll":0,"Yaw":0 }
   "cam_right": { "X": 1.5, "Y":  0.1, "Z": -0.8, "Pitch":0,"Roll":0,"Yaw":0 }

 Uso:
   Avvia il simulatore FSDS, poi esegui questo script.
   Premi 'q' nella finestra video per uscire.
=============================================================================
"""

import sys
import os

# Aggiunge la directory padre (python/) al path, cosi Python trova il package fsds
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import fsds
import cv2
import numpy as np

# =============================================================================
# CONNESSIONE AL SIMULATORE
# =============================================================================

print("[INFO] Connessione al simulatore FSDS...")
client = fsds.FSDSClient()

# Verifica connessione: se il ping fallisce il programma esce automaticamente
client.confirmConnection()
print("[INFO] Connesso! Avvio flusso video stereo...")

# enableApiControl e necessario per ricevere dati dai sensori via API
client.enableApiControl(True)


# =============================================================================
# FUNZIONE DI DECODIFICA IMMAGINE  <-- Punto critico, bug frequente!
# =============================================================================

def decode_image(response: fsds.ImageResponse) -> np.ndarray:
    """
    Converte un ImageResponse di FSDS/AirSim in un array NumPy BGR per OpenCV.

    PERCHE' E' COMPLICATO:
    - Con compress=True (default):  il simulatore restituisce un PNG/JPEG compresso.
      I byte sono in response.image_data_uint8 come stringa di bytes.
      Si usa cv2.imdecode() per decomprimere e ottenere direttamente un array BGR.

    - Con compress=False:  il simulatore restituisce pixel grezzi in formato RGBA
      (4 canali, non 3!). AirSim usa internamente RGBA. Bisogna:
        1. Leggere i byte con np.frombuffer()
        2. Fare reshape in (height, width, 4)
        3. Convertire RGBA -> BGR eliminando il canale Alpha

    In questo script usiamo compress=True perche':
      - cv2.imdecode e' piu' semplice e meno soggetto a errori di reshape
      - La compressione PNG e' lossless, quindi non perdiamo qualita'
      - E' piu' robusto quando width/height cambiano dinamicamente

    ATTENZIONE: image_data_uint8 e' un oggetto di tipo bytes.
    np.frombuffer(..., dtype=np.uint8) lo interpreta come sequenza di byte grezzi.
    """
    # Converte i byte compressi (PNG) in un array 1D di uint8
    img_array_1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)

    # cv2.imdecode decomprime il PNG e restituisce direttamente un array HxWx3 in BGR
    # IMREAD_COLOR forza sempre 3 canali BGR, anche se l'immagine sorgente fosse in scala di grigi
    img_bgr = cv2.imdecode(img_array_1d, cv2.IMREAD_COLOR)

    if img_bgr is None:
        # Questo accade se i byte sono corrotti o se il simulatore ha restituito un frame vuoto
        raise ValueError(
            "cv2.imdecode ha restituito None per la camera '{}'.".format(response.camera_name) +
            " Verifica che la camera sia definita in settings.json e che il simulatore sia in esecuzione."
        )

    return img_bgr


# =============================================================================
# CICLO PRINCIPALE - FLUSSO VIDEO IN TEMPO REALE
# =============================================================================

print("[INFO] Flusso video avviato. Premi 'q' nella finestra per uscire.")

while True:

    # -------------------------------------------------------------------------
    # ACQUISIZIONE SINCRONA CON UNA SINGOLA CHIAMATA RPC
    # -------------------------------------------------------------------------
    # FONDAMENTALE: le due ImageRequest vengono passate insieme in UNA sola
    # chiamata simGetImages(). Il server le esegue nello stesso tick di simulazione,
    # garantendo la sincronizzazione temporale tra i due frame.
    #
    # Se facessimo due chiamate separate:
    #   img_l = client.simGetImages([req_left])   # tick N
    #   img_r = client.simGetImages([req_right])  # tick N+1 (o N+k)
    # i frame sarebbero sfasati nel tempo a causa della latenza di rete (5-20ms
    # per chiamata) e del ciclo di rendering del simulatore.
    # Questo sfasamento causa artefatti gravi in qualsiasi algoritmo stereo.
    # -------------------------------------------------------------------------
    responses = client.simGetImages(
        [
            fsds.ImageRequest(
                camera_name='cam_left',
                image_type=fsds.ImageType.Scene,
                pixels_as_float=False,  # vogliamo uint8, non float32
                compress=True           # PNG compresso -> usiamo cv2.imdecode
            ),
            fsds.ImageRequest(
                camera_name='cam_right',
                image_type=fsds.ImageType.Scene,
                pixels_as_float=False,
                compress=True
            ),
        ],
        vehicle_name='FSCar'
    )

    # Verifica che il simulatore abbia restituito esattamente 2 risposte
    if len(responses) < 2:
        print("[WARN] Risposta incompleta dal simulatore, frame saltato.")
        continue

    response_left, response_right = responses[0], responses[1]

    # Salta il frame se uno dei due e' vuoto (es. simulatore in pausa o in caricamento)
    if len(response_left.image_data_uint8) == 0 or len(response_right.image_data_uint8) == 0:
        print("[WARN] Frame vuoto ricevuto, attendo il prossimo tick...")
        continue

    # -------------------------------------------------------------------------
    # DECODIFICA E CONVERSIONE IN ARRAY NUMPY
    # -------------------------------------------------------------------------
    try:
        frame_left  = decode_image(response_left)
        frame_right = decode_image(response_right)
    except ValueError as e:
        print("[ERROR] {}".format(e))
        continue

    # -------------------------------------------------------------------------
    # COMPOSIZIONE DEL FRAME STEREO SIDE-BY-SIDE
    # -------------------------------------------------------------------------
    # numpy.hstack impila orizzontalmente i due array.
    # Requisito: devono avere la stessa height (numero di righe) e lo stesso dtype.
    # Se usi risoluzioni diverse per le due camere, decommenta la riga seguente:
    #   frame_right = cv2.resize(frame_right, (frame_left.shape[1], frame_left.shape[0]))
    stereo_frame = np.hstack((frame_left, frame_right))

    # Aggiunge una linea verticale centrale come separatore visivo
    center_x = stereo_frame.shape[1] // 2
    cv2.line(stereo_frame, (center_x, 0), (center_x, stereo_frame.shape[0]), (0, 255, 0), 1)

    # Label sovrapposte per identificare le due viste
    cv2.putText(stereo_frame, "LEFT",  (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(stereo_frame, "RIGHT", (center_x + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(stereo_frame, "{}x{}".format(frame_left.shape[1], frame_left.shape[0]),
                (10, stereo_frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # -------------------------------------------------------------------------
    # VISUALIZZAZIONE
    # -------------------------------------------------------------------------
    cv2.imshow("Stereo Vision FSDS", stereo_frame)

    # waitKey(1) attende 1ms il tasto premuto ed e' necessario per aggiornare
    # la finestra OpenCV; senza di essa la finestra sarebbe congelata.
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("[INFO] Tasto 'q' premuto. Uscita in corso...")
        break

# =============================================================================
# CLEANUP
# =============================================================================
cv2.destroyAllWindows()
client.enableApiControl(False)
print("[INFO] Script terminato correttamente.")
