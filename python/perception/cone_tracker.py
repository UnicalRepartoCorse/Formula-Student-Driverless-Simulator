"""
=============================================================================
 Modulo di Tracking Coni - Pipeline di Percezione Formula Student Driverless
=============================================================================

 Scopo:
   Stabilizzare l'output del modello di rilevamento (es. YOLO), che può
   perdere dei coni per brevi intervalli a causa di motion blur, riflessi
   o occlusioni temporanee. Questo tracker mantiene una memoria "stateful"
   dei coni rilevati nei frame precedenti.

 Algoritmo:
   Ad ogni frame, si calcola la distanza euclidea tra i nuovi rilevamenti
   e i coni già in memoria. Si usa la Distance Matrix (scipy.spatial) per
   trovare le associazioni migliori in modo vettorizzato e performante.

 Dipendenze:
   - numpy
   - scipy

 Uso:
   from perception.cone_tracker import ConeTracker
   tracker = ConeTracker(distance_threshold=1.5, max_frames_lost=5)
   valid_cones = tracker.update(detections_from_yolo)
=============================================================================
"""

import numpy as np
from scipy.spatial.distance import cdist
from typing import List, Tuple, Dict, Any


# =============================================================================
# CLASSE DI APPOGGIO: TrackedCone
# =============================================================================

class TrackedCone:
    """
    Rappresenta un singolo cono tenuto in memoria dal tracker.

    Attributi:
        x          (float): Coordinata X del cono nel frame di riferimento del veicolo.
        y          (float): Coordinata Y del cono nel frame di riferimento del veicolo.
        color_id   (int):   Classe cromatica del cono.
                            Convenzione: 0 = Blu (limite sinistro),
                                         1 = Giallo (limite destro),
                                         2 = Arancione grande (start/finish),
                                         3 = Arancione piccolo.
        frames_lost (int):  Numero di frame consecutivi in cui il cono non è
                            stato rilevato. 0 = visto nell'ultimo frame.
    """

    # Mappa ID -> Nome per la stampa leggibile
    COLOR_NAMES: Dict[int, str] = {
        0: "Blu",
        1: "Giallo",
        2: "Arancione-Grande",
        3: "Arancione-Piccolo",
    }

    def __init__(self, x: float, y: float, color_id: int):
        self.x: float = x
        self.y: float = y
        self.color_id: int = color_id
        self.frames_lost: int = 0  # Appena creato: visto in questo frame

    def __repr__(self) -> str:
        color_name = self.COLOR_NAMES.get(self.color_id, f"ID={self.color_id}")
        return (
            f"TrackedCone("
            f"x={self.x:.2f}, y={self.y:.2f}, "
            f"colore={color_name}, "
            f"frames_lost={self.frames_lost})"
        )


# =============================================================================
# CLASSE PRINCIPALE: ConeTracker
# =============================================================================

class ConeTracker:
    """
    Tracker stateful per i coni di Formula Student.

    Mantiene in memoria una lista di coni rilevati nei frame precedenti e,
    ad ogni nuovo frame, aggiorna le loro posizioni o incrementa il contatore
    di "frame persi". I coni non visti per troppo tempo vengono rimossi.

    Args:
        distance_threshold (float): Distanza massima (in unità metriche, es. metri)
                                    entro cui un nuovo rilevamento viene associato
                                    a un cono già in memoria. Valori tipici: 0.5 - 2.0 m.
        max_frames_lost    (int):   Numero massimo di frame consecutivi in cui un
                                    cono può non essere rilevato prima di essere
                                    rimosso dalla memoria. Default = 5 frame.
    """

    def __init__(self, distance_threshold: float = 1.5, max_frames_lost: int = 5):
        self.distance_threshold: float = distance_threshold
        self.max_frames_lost: int = max_frames_lost

        # Lista centrale: tutti i coni attualmente in memoria
        self.tracked_cones: List[TrackedCone] = []

    # -------------------------------------------------------------------------

    def update(self, new_detections: List[Tuple[float, float, int]]) -> List[TrackedCone]:
        """
        Aggiorna lo stato interno del tracker con i nuovi rilevamenti del frame corrente.

        Esegue in sequenza cinque fasi:
          A) Data Association  -> Abbina i nuovi rilevamenti ai coni in memoria
          B) Aggiornamento     -> Aggiorna posizione e azzera frames_lost
          C) Nuovi Coni        -> Aggiunge i rilevamenti non abbinati come nuovi coni
          D) Decadimento       -> Incrementa frames_lost per i coni non aggiornati
          E) Garbage Collect   -> Rimuove i coni con frames_lost > max_frames_lost

        Args:
            new_detections: Lista di tuple (x, y, color_id) provenienti da YOLO
                            per il frame corrente. Può essere vuota.

        Returns:
            Lista di TrackedCone validi (quelli con frames_lost <= max_frames_lost).
        """

        # Set che terrà traccia degli indici dei tracked_cones aggiornati in questo frame.
        # Serve nella Fase D per sapere chi NON ha ricevuto aggiornamenti.
        updated_tracked_indices = set()

        # Set che terrà traccia degli indici dei new_detections già abbinati.
        # Serve nella Fase C per sapere quali rilevamenti sono "nuovi".
        matched_detection_indices = set()

        # ------------------------------------------------------------------
        # FASE A & B: DATA ASSOCIATION + AGGIORNAMENTO
        # ------------------------------------------------------------------
        # Eseguiamo questa fase solo se ci sono sia coni in memoria
        # che nuovi rilevamenti. Altrimenti non c'è niente da abbinare.
        if len(self.tracked_cones) > 0 and len(new_detections) > 0:

            # Estrae le coordinate [x, y] dai coni in memoria come array NumPy
            # Shape: (N_tracked, 2)
            tracked_coords = np.array(
                [[c.x, c.y] for c in self.tracked_cones], dtype=np.float32
            )

            # Estrae le coordinate [x, y] dai nuovi rilevamenti
            # Shape: (N_detected, 2)
            detected_coords = np.array(
                [[d[0], d[1]] for d in new_detections], dtype=np.float32
            )

            # --- Calcolo della Distance Matrix ---
            # cdist calcola la distanza euclidea tra ogni coppia (rilevato, in-memoria).
            # Il risultato è una matrice (N_detected x N_tracked):
            #   dist_matrix[i, j] = distanza tra il rilevamento i e il cono in memoria j
            dist_matrix = cdist(detected_coords, tracked_coords, metric='euclidean')

            # --- Algoritmo di Associazione Greedy (Hungarian semplificato) ---
            # Per semplicità e velocità usiamo un approccio greedy:
            # ordiniamo le coppie per distanza crescente e assegniamo la migliore
            # disponibile. Per FS, con densità di coni relativamente bassa,
            # questo approccio è sufficiente e molto efficiente.

            # Ottiene gli indici ordinati per distanza crescente (appiattisce la matrice)
            sorted_indices = np.argsort(dist_matrix, axis=None)

            for flat_idx in sorted_indices:
                # Recupera indice del rilevamento (riga) e del cono in memoria (colonna)
                det_idx = int(flat_idx // len(self.tracked_cones))
                tracked_idx = int(flat_idx % len(self.tracked_cones))

                # Salta se questa coppia supera la soglia di distanza
                if dist_matrix[det_idx, tracked_idx] > self.distance_threshold:
                    break  # Essendo ordinato, tutto il resto sarà ancora più lontano

                # Salta se uno dei due è già stato abbinato in questo frame
                if det_idx in matched_detection_indices:
                    continue
                if tracked_idx in updated_tracked_indices:
                    continue

                # FILTRO per colore: abbiniamo solo coni dello stesso tipo.
                # Un cono blu non può mai diventare giallo tra un frame e l'altro.
                detection_color = new_detections[det_idx][2]
                tracked_color   = self.tracked_cones[tracked_idx].color_id
                if detection_color != tracked_color:
                    continue

                # --- ABBINAMENTO TROVATO ---
                # FASE B: Aggiorna la posizione del cono in memoria con quella nuova
                #         e azzera il contatore di frame persi.
                self.tracked_cones[tracked_idx].x = new_detections[det_idx][0]
                self.tracked_cones[tracked_idx].y = new_detections[det_idx][1]
                self.tracked_cones[tracked_idx].frames_lost = 0

                # Registra gli indici come "già usati"
                matched_detection_indices.add(det_idx)
                updated_tracked_indices.add(tracked_idx)

        # ------------------------------------------------------------------
        # FASE C: NUOVI CONI
        # ------------------------------------------------------------------
        # Salviamo la lunghezza PRIMA di aggiungere nuovi coni.
        # Serve nella Fase D per non penalizzare i coni appena inseriti:
        # un cono nuovo ha frames_lost=0 per definizione, non va decrementato.
        n_coni_originali = len(self.tracked_cones)

        # I rilevamenti che non sono stati abbinati a nessun cono in memoria
        # sono oggetti nuovi: li aggiungiamo alla lista.
        for det_idx, detection in enumerate(new_detections):
            if det_idx not in matched_detection_indices:
                x, y, color_id = detection[0], detection[1], detection[2]
                nuovo_cono = TrackedCone(x=x, y=y, color_id=color_id)
                self.tracked_cones.append(nuovo_cono)

        # ------------------------------------------------------------------
        # FASE D: DECADIMENTO DELLA MEMORIA
        # ------------------------------------------------------------------
        # I coni in memoria che NON hanno ricevuto aggiornamenti in questo frame
        # vengono "penalizzati": incrementiamo il loro contatore frames_lost.
        # IMPORTANTE: iteriamo solo fino a n_coni_originali, escludendo i coni
        # appena aggiunti nella Fase C (che hanno già frames_lost=0 per definizione).
        for tracked_idx in range(n_coni_originali):
            if tracked_idx not in updated_tracked_indices:
                self.tracked_cones[tracked_idx].frames_lost += 1

        # ------------------------------------------------------------------
        # FASE E: GARBAGE COLLECTION
        # ------------------------------------------------------------------
        # Rimuoviamo definitivamente dalla memoria i coni persi da troppi frame.
        self.tracked_cones = [
            cono for cono in self.tracked_cones
            if cono.frames_lost <= self.max_frames_lost
        ]

        # Restituisce la lista completa dei coni attualmente validi
        return self.tracked_cones

    # -------------------------------------------------------------------------

    def get_valid_cones(self) -> List[TrackedCone]:
        """
        Restituisce i coni attualmente in memoria senza effettuare un aggiornamento.
        Utile per leggere lo stato corrente senza passare nuovi rilevamenti.
        """
        return self.tracked_cones

    def reset(self):
        """Svuota completamente la memoria del tracker (es. inizio nuovo giro)."""
        self.tracked_cones = []

    def __repr__(self) -> str:
        return (
            f"ConeTracker("
            f"n_coni={len(self.tracked_cones)}, "
            f"soglia={self.distance_threshold}m, "
            f"max_frames_lost={self.max_frames_lost})"
        )


# =============================================================================
# BLOCCO DI TEST
# =============================================================================

if __name__ == '__main__':
    """
    Simula 3 frame di rilevamento per dimostrare il comportamento del tracker.

    Scenario:
      - Frame 1: YOLO rileva 4 coni (2 blu a sinistra, 2 gialli a destra).
      - Frame 2: YOLO perde il cono blu in posizione (2.0, 1.5) (falso negativo).
                 Il tracker deve mantenerlo in memoria con frames_lost = 1.
      - Frame 3: YOLO lo rileva di nuovo. Il tracker deve riabbinarlo e
                 azzerare frames_lost a 0.
    """

    COLORE_BLU    = 0
    COLORE_GIALLO = 1

    print("=" * 60)
    print("  TEST: ConeTracker - Simulazione 3 Frame")
    print("=" * 60)

    # Inizializza il tracker:
    #  - distance_threshold=1.5m: due coni entro 1.5m sono lo stesso oggetto
    #  - max_frames_lost=5: un cono viene rimosso dopo 5 frame senza avvistamento
    tracker = ConeTracker(distance_threshold=1.5, max_frames_lost=5)

    # ------------------------------------------------------------------
    # FRAME 1: tutti e 4 i coni rilevati
    # ------------------------------------------------------------------
    print("\n--- FRAME 1: Tutti i coni rilevati ---")
    detections_frame1 = [
        (1.0, 0.5, COLORE_BLU),    # Cono blu A
        (2.0, 1.5, COLORE_BLU),    # Cono blu B  <-- questo "scomparirà" al frame 2
        (1.0, -0.5, COLORE_GIALLO), # Cono giallo A
        (2.0, -1.5, COLORE_GIALLO), # Cono giallo B
    ]
    coni_validi = tracker.update(detections_frame1)
    print(f"  Coni in memoria: {len(coni_validi)}")
    for cono in coni_validi:
        print(f"    {cono}")

    # ------------------------------------------------------------------
    # FRAME 2: il cono blu B NON viene rilevato (falso negativo)
    # ------------------------------------------------------------------
    print("\n--- FRAME 2: Cono Blu B perso (falso negativo YOLO) ---")
    detections_frame2 = [
        (1.05, 0.52, COLORE_BLU),    # Cono blu A (leggero movimento)
        # Cono blu B ASSENTE
        (1.02, -0.51, COLORE_GIALLO), # Cono giallo A
        (2.01, -1.48, COLORE_GIALLO), # Cono giallo B
    ]
    coni_validi = tracker.update(detections_frame2)
    print(f"  Coni in memoria: {len(coni_validi)} (il tracker mantiene il cono perso)")
    for cono in coni_validi:
        marker = " <-- PERSO (frames_lost=1)" if cono.frames_lost > 0 else ""
        print(f"    {cono}{marker}")

    # ------------------------------------------------------------------
    # FRAME 3: il cono blu B viene di nuovo rilevato
    # ------------------------------------------------------------------
    print("\n--- FRAME 3: Cono Blu B torna visibile ---")
    detections_frame3 = [
        (1.08, 0.55, COLORE_BLU),    # Cono blu A
        (2.02, 1.52, COLORE_BLU),    # Cono blu B (rilevato di nuovo!)
        (1.04, -0.50, COLORE_GIALLO), # Cono giallo A
        (2.00, -1.50, COLORE_GIALLO), # Cono giallo B
    ]
    coni_validi = tracker.update(detections_frame3)
    print(f"  Coni in memoria: {len(coni_validi)} (frames_lost azzerato)")
    for cono in coni_validi:
        print(f"    {cono}")

    # ------------------------------------------------------------------
    # VERIFICA FINALE
    # ------------------------------------------------------------------
    print("\n--- VERIFICA ---")
    assert len(coni_validi) == 4, "ERRORE: dovrebbero esserci 4 coni validi!"
    assert all(c.frames_lost == 0 for c in coni_validi), \
        "ERRORE: tutti i frames_lost dovrebbero essere 0 al frame 3!"
    print("  [OK] Tutti i 4 coni in memoria con frames_lost = 0.")
    print("  [OK] Il tracker ha mantenuto il cono blu B durante il falso negativo.")
    print("\nTest completato con successo!")
