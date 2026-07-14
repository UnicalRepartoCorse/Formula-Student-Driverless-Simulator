"""
=============================================================================
 Modulo di Controllo - Pure Pursuit Controller
 Pipeline di Controllo per Formula Student Driverless
=============================================================================

 Scopo:
   Calcola il comando di sterzo normalizzato per seguire la centerline
   usando l'algoritmo geometrico Pure Pursuit.

 Algoritmo Pure Pursuit:
   1. Seleziona un "target point" sulla centerline a distanza lookahead_distance
   2. Calcola l'angolo alpha verso il target
   3. Applica la formula: steering = arctan(2 * L * sin(alpha) / ld)
   4. Normalizza nel range [-1.0, 1.0] richiesto dal simulatore

 Sistema di Riferimento (convenzione FSDS):
   Origine (0, 0) = posizione della vettura
   X = asse laterale    (positivo a destra)
   Z = asse di avanzamento (positivo in avanti)

 Dipendenze:
   - numpy
   - matplotlib (solo per il blocco di test)

 Uso:
   from perception.pure_pursuit import PurePursuitController
   controller = PurePursuitController(wheelbase=1.53, lookahead_distance=4.0)
   steering_cmd = controller.calculate_steering(centerline)
=============================================================================
"""

import numpy as np
import math
from typing import List, Tuple, Optional


# Tipo alias per leggibilità
Point = Tuple[float, float]  # (x, z)


# =============================================================================
# CLASSE PRINCIPALE: PurePursuitController
# =============================================================================

class PurePursuitController:
    """
    Implementa l'algoritmo Pure Pursuit per il controllo dello sterzo.

    Il Pure Pursuit è un algoritmo geometrico classico nella guida autonoma:
    invece di seguire il punto immediatamente davanti, "mira" a un punto
    sulla traiettoria a una certa distanza (lookahead_distance), calcolando
    lo sterzo necessario per raggiungerlo su un arco di cerchio.

    Maggiore è la lookahead_distance → sterzo più fluido ma meno reattivo.
    Minore è la lookahead_distance → sterzo più aggressivo ma instabile.

    Args:
        wheelbase         (float): Passo dell'auto (distanza asse ant. - post.) in metri.
                                   Default = 1.53 m (tipico FS).
        lookahead_distance (float): Distanza di mira in metri. Default = 4.0 m.
        max_steering_angle (float): Angolo massimo delle ruote in radianti.
                                    Default = 0.52 rad (~30°).
    """

    def __init__(
        self,
        wheelbase: float = 1.53,
        lookahead_distance: float = 4.0,
        max_steering_angle: float = 0.52
    ):
        self.wheelbase = wheelbase
        self.lookahead_distance = lookahead_distance
        self.max_steering_angle = max_steering_angle

    # -------------------------------------------------------------------------

    def _find_target_point(self, centerline: List[Point]) -> Point:
        """
        Fase A - Ricerca del Target Point.

        Scorre la centerline ordinata per Z crescente e restituisce
        il primo punto la cui distanza euclidea dall'origine (0,0)
        è >= lookahead_distance.

        Se tutti i punti sono più vicini del lookahead (centerline corta
        o siamo già oltre i coni), restituisce l'ultimo punto disponibile
        come fallback sicuro.

        Args:
            centerline: Lista di punti (x, z) ordinati per Z crescente.

        Returns:
            Il punto target (x, z) selezionato.
        """
        for punto in centerline:
            x, z = punto[0], punto[1]
            distanza = math.sqrt(x ** 2 + z ** 2)
            if distanza >= self.lookahead_distance:
                return punto  # Primo punto oltre la soglia

        # Fallback: la centerline è più corta del lookahead → ultimo punto
        return centerline[-1]

    # -------------------------------------------------------------------------

    def calculate_steering(self, centerline: List[Point]) -> float:
        """
        Calcola il comando di sterzo normalizzato tramite Pure Pursuit.

        Fasi:
          A) Trova il target point sulla centerline
          B) Calcola l'angolo alpha verso il target
          C) Applica la formula del Pure Pursuit
          D) Normalizza nel range [-1.0, 1.0]

        Args:
            centerline: Lista di punti [(x1,z1), (x2,z2), ...] ordinati
                        per Z crescente (output del PathPlanner).

        Returns:
            Comando di sterzo normalizzato in [-1.0, 1.0].
            Negativo = sterza a sinistra, Positivo = sterza a destra.
            Restituisce 0.0 se la centerline è vuota.
        """

        # Gestione centerline vuota: nessun input → vai dritto
        if not centerline:
            return 0.0

        # ------------------------------------------------------------------
        # FASE A: Ricerca del Target Point
        # ------------------------------------------------------------------
        target_point = self._find_target_point(centerline)
        x_target = target_point[0]
        z_target = target_point[1]

        # ------------------------------------------------------------------
        # FASE B: Calcolo dell'Angolo Alpha
        # ------------------------------------------------------------------
        # Alpha è l'angolo tra l'asse di avanzamento Z (direzione dritto)
        # e il vettore che punta al target_point.
        #
        # math.atan2(x, z) restituisce l'angolo in radianti:
        #   - atan2(x>0, z) → alpha positivo → target a DESTRA
        #   - atan2(x<0, z) → alpha negativo → target a SINISTRA
        #
        # NOTA: usiamo atan2(x, z) e NON atan2(z, x) perché il nostro
        # asse di riferimento è Z (avanzamento), non X (laterale).
        # Questo è il tipico bug da invertire in un sistema X-Z vs X-Y.
        alpha = math.atan2(x_target, z_target)

        # ------------------------------------------------------------------
        # FASE C: Formula Pure Pursuit
        # ------------------------------------------------------------------
        # La formula deriva dalla geometria del cerchio di curvatura:
        #   steering_angle = arctan( (2 * L * sin(alpha)) / ld )
        #
        # Dove:
        #   L  = wheelbase (passo del veicolo)
        #   ld = lookahead_distance (distanza di mira)
        #
        # Il termine (2 * L * sin(alpha) / ld) è la curvatura necessaria.
        # arctan la converte nell'angolo effettivo dello sterzo.
        steering_angle = math.atan2(
            2.0 * self.wheelbase * math.sin(alpha),
            self.lookahead_distance
        )

        # ------------------------------------------------------------------
        # FASE D: Normalizzazione nel range [-1.0, 1.0]
        # ------------------------------------------------------------------
        # Il simulatore FSDS si aspetta un valore normalizzato, non radianti.
        # Dividiamo per max_steering_angle per scalare il range:
        #   steering_angle == +max_steering_angle → output = +1.0 (pieno a destra)
        #   steering_angle == -max_steering_angle → output = -1.0 (pieno a sinistra)
        #
        # numpy.clip garantisce che valori oltre il range (es. in curve strette)
        # vengano saturati a ±1.0 senza causare errori nel simulatore.
        steering_normalized = steering_angle / self.max_steering_angle
        steering_normalized = float(np.clip(steering_normalized, -1.0, 1.0))

        return steering_normalized

    # -------------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"PurePursuitController("
            f"wheelbase={self.wheelbase}m, "
            f"lookahead={self.lookahead_distance}m, "
            f"max_steer={math.degrees(self.max_steering_angle):.1f}°)"
        )


# =============================================================================
# BLOCCO DI TEST E VISUALIZZAZIONE
# =============================================================================

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    print("=" * 60)
    print("  TEST: PurePursuitController")
    print("=" * 60)

    controller = PurePursuitController(
        wheelbase=1.53,
        lookahead_distance=4.0,
        max_steering_angle=0.52
    )
    print(f"\n  Controller: {controller}")

    # ------------------------------------------------------------------
    # Tre scenari di test con centerline sintetiche
    # ------------------------------------------------------------------
    scenari = [
        {
            "nome":       "Rettilineo (centerline centrata)",
            "centerline": [(0.0, 2.0), (0.0, 4.0), (0.0, 6.0), (0.0, 8.0)],
            "atteso":     "≈ 0.0 (nessuno sterzo)",
        },
        {
            "nome":       "Curva destra (target a destra)",
            "centerline": [(1.0, 2.0), (2.0, 4.0), (3.0, 5.5), (4.0, 7.0)],
            "atteso":     "> 0.0 (sterza a destra)",
        },
        {
            "nome":       "Curva sinistra (target a sinistra)",
            "centerline": [(-1.0, 2.0), (-2.0, 4.0), (-3.0, 5.5), (-4.0, 7.0)],
            "atteso":     "< 0.0 (sterza a sinistra)",
        },
    ]

    risultati = []
    print()
    for s in scenari:
        cmd = controller.calculate_steering(s["centerline"])
        risultati.append(cmd)
        print(f"  Scenario: {s['nome']}")
        print(f"    Atteso:   {s['atteso']}")
        print(f"    Output:   {cmd:+.4f}")
        print()

    # Verifica logica
    assert abs(risultati[0]) < 0.05,  "ERRORE: rettilineo dovrebbe dare ~0"
    assert risultati[1] > 0,          "ERRORE: curva destra dovrebbe dare > 0"
    assert risultati[2] < 0,          "ERRORE: curva sinistra dovrebbe dare < 0"
    print("  [OK] Tutti gli scenari verificati correttamente.\n")

    # ------------------------------------------------------------------
    # Simulazione dinamica: la vettura percorre una curva
    # e l'output dello sterzo viene registrato frame per frame
    # ------------------------------------------------------------------
    print("  Simulazione dinamica su tracciato curvo...")

    # Centerline che simula una curva progressiva a destra
    n_punti = 10
    centerline_curva = [
        (0.3 * i ** 1.3, i * 2.5)
        for i in range(1, n_punti + 1)
    ]

    comandi_sterzo = []
    lookahead_range = np.linspace(1.5, 7.0, 50)

    for ld in lookahead_range:
        ctrl_temp = PurePursuitController(
            wheelbase=1.53,
            lookahead_distance=float(ld),
            max_steering_angle=0.52
        )
        cmd = ctrl_temp.calculate_steering(centerline_curva)
        comandi_sterzo.append(cmd)

    # ------------------------------------------------------------------
    # Visualizzazione
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
    fig.patch.set_facecolor('#16213e')

    # --- GRAFICO 1: Tracciato e target point ---
    ax1.set_facecolor('#1a1a2e')

    # Disegna la centerline curva
    cx = [p[0] for p in centerline_curva]
    cz = [p[1] for p in centerline_curva]
    ax1.plot(cx, cz, 'o--', color='#00e676', linewidth=2, markersize=6,
             label='Centerline', zorder=4)

    # Evidenzia il target point con lookahead=4.0
    target = controller._find_target_point(centerline_curva)
    ax1.scatter([target[0]], [target[1]], color='#ff6b6b', s=180, zorder=6,
                label=f'Target Point (ld={controller.lookahead_distance}m)', marker='*')

    # Cerchio di lookahead centrato sulla vettura
    cerchio = plt.Circle((0, 0), controller.lookahead_distance,
                          color='#ffd700', fill=False, linestyle='--',
                          linewidth=1.2, alpha=0.6, label=f'Cerchio lookahead ({controller.lookahead_distance}m)')
    ax1.add_patch(cerchio)

    # Freccia dalla vettura al target
    ax1.annotate(
        '', xy=(target[0], target[1]), xytext=(0, 0),
        arrowprops=dict(arrowstyle='->', color='#ff6b6b', lw=1.5)
    )

    # Vettura all'origine
    ax1.scatter([0], [0], color='white', s=200, marker='D', zorder=7, label='Vettura')

    cmd_finale = controller.calculate_steering(centerline_curva)
    ax1.set_title(
        f'Pure Pursuit — Tracciato e Target\nSterzo calcolato: {cmd_finale:+.3f}',
        color='white', fontweight='bold'
    )
    ax1.set_xlabel('X — Laterale (m)', color='white')
    ax1.set_ylabel('Z — Avanzamento (m)', color='white')
    ax1.tick_params(colors='white')
    for spine in ax1.spines.values():
        spine.set_color('#333366')
    ax1.legend(facecolor='#1a1a2e', edgecolor='white', labelcolor='white', fontsize=8)
    ax1.set_aspect('equal', adjustable='box')
    ax1.grid(True, color='white', alpha=0.06)

    # --- GRAFICO 2: Sterzo vs Lookahead Distance ---
    ax2.set_facecolor('#1a1a2e')
    ax2.plot(lookahead_range, comandi_sterzo, color='#4e9af1', linewidth=2.5)
    ax2.axhline(y=0, color='white', linestyle='--', linewidth=0.8, alpha=0.4)
    ax2.axhline(y=1.0, color='#ff6b6b', linestyle=':', linewidth=1.0, alpha=0.7, label='Saturazione ±1.0')
    ax2.axhline(y=-1.0, color='#ff6b6b', linestyle=':', linewidth=1.0, alpha=0.7)
    ax2.axvline(x=4.0, color='#ffd700', linestyle='--', linewidth=1.2, alpha=0.8, label='ld default = 4.0m')
    ax2.fill_between(lookahead_range, comandi_sterzo, 0,
                     where=[c > 0 for c in comandi_sterzo],
                     alpha=0.15, color='#4e9af1')
    ax2.set_title('Sensibilità: Sterzo vs Lookahead Distance\n(stessa curva, lookahead variabile)',
                  color='white', fontweight='bold')
    ax2.set_xlabel('Lookahead Distance (m)', color='white')
    ax2.set_ylabel('Comando Sterzo Normalizzato [-1, 1]', color='white')
    ax2.tick_params(colors='white')
    for spine in ax2.spines.values():
        spine.set_color('#333366')
    ax2.legend(facecolor='#1a1a2e', edgecolor='white', labelcolor='white', fontsize=8)
    ax2.set_ylim(-1.2, 1.2)
    ax2.grid(True, color='white', alpha=0.06)

    plt.suptitle('Pure Pursuit Controller — Formula Student Driverless',
                 color='white', fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    print("  Grafico generato. Chiudi la finestra per terminare.")
    plt.show()

    print("Test completato con successo!")
