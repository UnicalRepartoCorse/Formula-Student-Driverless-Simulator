"""
=============================================================================
 Modulo di Path Planning - Calcolo Centerline
 Pipeline di Pianificazione per Formula Student Driverless
=============================================================================

 Scopo:
   A partire dalle posizioni dei coni blu (sinistra) e gialli (destra)
   fornite dal modulo di percezione, calcola la traiettoria ideale (centerline)
   come sequenza di punti medi tra le coppie di coni opposte.

 Algoritmo - Midpoint Matching:
   Per ogni cono blu, si trova il cono giallo euclideamente più vicino.
   Il punto medio della coppia (blu, giallo) è un punto della centerline.
   La lista risultante viene ordinata per Z crescente (dal più vicino
   alla vettura al più lontano) per formare un percorso sequenziale.

 Sistema di Riferimento (convenzione FSDS):
   X = asse laterale   (positivo a destra)
   Z = asse di avanzamento (positivo in avanti, davanti alla vettura)

 Dipendenze:
   - numpy
   - matplotlib (solo per il blocco di test)

 Uso:
   from perception.path_planner import PathPlanner
   planner = PathPlanner()
   centerline = planner.compute_centerline(blue_cones, yellow_cones)
=============================================================================
"""

import numpy as np
import math
from typing import List, Tuple


# Tipo alias per leggibilità
Cone  = Tuple[float, float]   # (x, z)
Point = Tuple[float, float]   # (x, z)


# =============================================================================
# CLASSE PRINCIPALE: PathPlanner
# =============================================================================

class PathPlanner:
    """
    Calcola la centerline ideale di un tracciato Formula Student
    a partire dalle posizioni dei coni di delimitazione rilevati.

    La centerline è la sequenza di punti medi tra ogni coppia
    (cono blu, cono giallo più vicino), ordinata per Z crescente.
    """

    # -------------------------------------------------------------------------

    @staticmethod
    def _distanza_euclidea(p1: Cone, p2: Cone) -> float:
        """
        Calcola la distanza euclidea tra due punti 2D.

        Args:
            p1: Primo punto (x, z).
            p2: Secondo punto (x, z).

        Returns:
            Distanza euclidea in metri.
        """
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    # -------------------------------------------------------------------------

    def compute_centerline(
        self,
        blue_cones: List[Cone],
        yellow_cones: List[Cone]
    ) -> List[Point]:
        """
        Calcola la centerline tramite algoritmo Midpoint Matching.

        Algoritmo:
          1. Per ogni cono blu, cerca il cono giallo euclideamente più vicino.
          2. Calcola il punto medio della coppia (blu, giallo).
          3. Aggiunge il midpoint alla lista centerline.
          4. Ordina la centerline per Z crescente (vicino → lontano).

        NOTA: Un cono giallo può essere abbinato a più coni blu
        (non c'è esclusione reciproca). In contesti con coni molto densi
        si può applicare un matching esclusivo (Hungarian algorithm),
        ma per FS il Nearest-Neighbor è generalmente sufficiente.

        Args:
            blue_cones:   Lista di (x, z) dei coni blu   (delimitazione sinistra).
            yellow_cones: Lista di (x, z) dei coni gialli (delimitazione destra).

        Returns:
            Lista di punti (x, z) che formano la centerline,
            ordinati per Z crescente. Lista vuota se l'input è insufficiente.
        """

        # Controllo dati minimi: servono almeno un cono per lato
        if not blue_cones or not yellow_cones:
            print("[WARN] compute_centerline: lista coni vuota, centerline non calcolabile.")
            return []

        centerline: List[Point] = []

        # ------------------------------------------------------------------
        # Fase 1 - Midpoint Matching
        # Iteriamo sui coni blu come coni "guida" e cerchiamo il giallo
        # più vicino per ciascuno.
        # ------------------------------------------------------------------
        for cono_blu in blue_cones:

            # Inizializza la ricerca del minimo
            distanza_minima = float('inf')
            cono_giallo_piu_vicino: Cone = yellow_cones[0]

            for cono_giallo in yellow_cones:
                d = self._distanza_euclidea(cono_blu, cono_giallo)
                if d < distanza_minima:
                    distanza_minima = d
                    cono_giallo_piu_vicino = cono_giallo

            # ------------------------------------------------------------------
            # Fase 2 - Calcolo del Midpoint
            # Il punto medio tra il cono blu e il cono giallo abbinato
            # è il punto ideale della traiettoria a quella profondità.
            # ------------------------------------------------------------------
            mid_x = (cono_blu[0] + cono_giallo_piu_vicino[0]) / 2.0
            mid_z = (cono_blu[1] + cono_giallo_piu_vicino[1]) / 2.0

            centerline.append((mid_x, mid_z))

        # ------------------------------------------------------------------
        # Fase 3 - Ordinamento per Z crescente
        # Ordiniamo i punti della centerline dal più vicino (Z piccola)
        # al più lontano (Z grande) rispetto alla vettura.
        # Questo garantisce che il controller riceva una sequenza di
        # waypoint nell'ordine corretto da seguire.
        # ------------------------------------------------------------------
        centerline.sort(key=lambda punto: punto[1])

        return centerline


# =============================================================================
# BLOCCO DI TEST E VISUALIZZAZIONE
# =============================================================================

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    print("=" * 60)
    print("  TEST: PathPlanner - Calcolo e Visualizzazione Centerline")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Generazione dati finti: simulazione di una curva verso destra
    #
    # Il tracciato curva dolcemente verso destra all'aumentare di Z.
    # I coni blu sono a sinistra (X negativo), i gialli a destra (X positivo).
    # Lo spostamento laterale dei coni cresce con Z per simulare la curva.
    # ------------------------------------------------------------------

    # Parametri del tracciato sintetico
    n_coni      = 6       # coppie di coni
    larghezza   = 3.5     # larghezza del tracciato in metri (tipica FS)
    passo_z     = 4.0     # distanza tra coppie successive sull'asse Z (metri)
    curvatura   = 0.18    # quanto curva il tracciato per ogni metro di Z

    blue_cones:   List[Cone] = []
    yellow_cones: List[Cone] = []

    for i in range(n_coni):
        z = i * passo_z  # profondità del cono i-esimo

        # Lo spostamento laterale del centro del tracciato aumenta con Z
        # simulando una curva verso destra (centro si sposta a destra = +X)
        centro_x = curvatura * z ** 1.4

        # I coni blu sono a sinistra del centro, i gialli a destra
        x_blu    = centro_x - larghezza / 2.0
        x_giallo = centro_x + larghezza / 2.0

        # Aggiunge un piccolo rumore per rendere il test più realistico
        rumore = 0.08
        x_blu    += np.random.uniform(-rumore, rumore)
        x_giallo += np.random.uniform(-rumore, rumore)
        z_con_rumore_blu    = z + np.random.uniform(-rumore / 2, rumore / 2)
        z_con_rumore_giallo = z + np.random.uniform(-rumore / 2, rumore / 2)

        blue_cones.append((x_blu, z_con_rumore_blu))
        yellow_cones.append((x_giallo, z_con_rumore_giallo))

    print(f"\n  Coni Blu    ({len(blue_cones)} totali):")
    for c in blue_cones:
        print(f"    x={c[0]:+.2f}  z={c[1]:.2f}")

    print(f"\n  Coni Gialli ({len(yellow_cones)} totali):")
    for c in yellow_cones:
        print(f"    x={c[0]:+.2f}  z={c[1]:.2f}")

    # ------------------------------------------------------------------
    # Calcolo Centerline
    # ------------------------------------------------------------------
    planner = PathPlanner()
    centerline = planner.compute_centerline(blue_cones, yellow_cones)

    print(f"\n  Centerline calcolata ({len(centerline)} waypoint):")
    for i, pt in enumerate(centerline):
        print(f"    Waypoint {i+1}: x={pt[0]:+.2f}  z={pt[1]:.2f}")

    # ------------------------------------------------------------------
    # Verifica di base
    # ------------------------------------------------------------------
    assert len(centerline) == len(blue_cones), \
        "ERRORE: il numero di waypoint deve essere uguale al numero di coni blu!"
    print("\n  [OK] Numero di waypoint corretto.")

    # Verifica che tutti i waypoint siano ordinati per Z crescente
    zs = [pt[1] for pt in centerline]
    assert zs == sorted(zs), "ERRORE: la centerline non è ordinata per Z crescente!"
    print("  [OK] Centerline ordinata correttamente per Z crescente.")

    # ------------------------------------------------------------------
    # Visualizzazione Matplotlib
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 10))
    ax.set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('#16213e')

    # Scatter: coni blu
    bx = [c[0] for c in blue_cones]
    bz = [c[1] for c in blue_cones]
    ax.scatter(bx, bz, color='#4e9af1', s=120, zorder=5, label='Coni Blu (sinistra)')

    # Scatter: coni gialli
    yx = [c[0] for c in yellow_cones]
    yz = [c[1] for c in yellow_cones]
    ax.scatter(yx, yz, color='#ffd700', s=120, zorder=5, label='Coni Gialli (destra)')

    # Icone triangolo per i coni (stile più realistico)
    for x, z in blue_cones:
        ax.plot(x, z, marker='^', markersize=14, color='#4e9af1', zorder=4)
    for x, z in yellow_cones:
        ax.plot(x, z, marker='^', markersize=14, color='#ffd700', zorder=4)

    # Linee tratteggiate che mostrano le coppie abbinate (blu → giallo)
    for i, cono_blu in enumerate(blue_cones):
        # Trova il giallo più vicino (stessa logica del planner)
        giallo_vicino = min(yellow_cones, key=lambda g: PathPlanner._distanza_euclidea(cono_blu, g))
        ax.plot(
            [cono_blu[0], giallo_vicino[0]],
            [cono_blu[1], giallo_vicino[1]],
            color='gray', linestyle='--', linewidth=0.8, alpha=0.5, zorder=3
        )

    # Plot: centerline
    cx = [pt[0] for pt in centerline]
    cz = [pt[1] for pt in centerline]
    ax.plot(cx, cz, color='#00e676', linewidth=2.5, marker='o',
            markersize=7, zorder=6, label='Centerline')

    # Waypoint numerati
    for i, (x, z) in enumerate(centerline):
        ax.annotate(
            f'W{i+1}',
            xy=(x, z), xytext=(x + 0.15, z + 0.05),
            color='#00e676', fontsize=8, fontweight='bold'
        )

    # Posizione della vettura all'origine
    ax.scatter([0], [0], color='white', s=200, marker='D', zorder=7, label='Vettura')
    ax.annotate('VETTURA', xy=(0, 0), xytext=(0.15, -0.4),
                color='white', fontsize=8, fontweight='bold')

    # Etichette e stile
    ax.set_xlabel('X  —  Asse Laterale (m)', color='white')
    ax.set_ylabel('Z  —  Asse di Avanzamento (m)', color='white')
    ax.set_title('PathPlanner — Centerline via Midpoint Matching\n(curva verso destra)',
                 color='white', fontsize=12, fontweight='bold')
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_color('#16213e')
    ax.spines['right'].set_color('#16213e')
    ax.legend(loc='upper left', facecolor='#1a1a2e', edgecolor='white', labelcolor='white')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, color='white', alpha=0.08)

    plt.tight_layout()
    print("\n  Grafico generato. Chiudi la finestra per terminare.")
    plt.show()

    print("Test completato con successo!")
