# Formula Student Driverless Simulator (FSDS) - ROS 2 Jazzy Stack

Questo README contiene le istruzioni e tutti i comandi necessari per configurare, compilare e avviare la pipeline di controllo a guida autonoma su WSL (Ubuntu) interfacciata con il simulatore FSDS su Windows.

---

## 1. Compilazione del Workspace

Spostati nella cartella principale del workspace ROS 2 su WSL ed esegui la build di tutti i pacchetti (compreso il modello 3D `eufs_racecar`, il bridge e i messaggi custom).
> **Tip:** Usa `--symlink-install` in modo che qualsiasi modifica successiva ai file sorgente Python venga applicata immediatamente senza dover rieseguire la build.

```bash
# Entra nella cartella di lavoro
cd /mnt/windows_path_to/Formula-Student-Driverless-Simulator/ros2

# Esegui il source di ROS 2 Jazzy
source /opt/ros/jazzy/setup.bash

# Compila tutti i pacchetti (driverless, fs_msgs, fsds_ros2_bridge)
colcon build --symlink-install

# Carica l'ambiente locale appena compilato
source install/setup.bash
```

---

## 2. Avvio del Bridge del Simulatore (FSDS ROS 2 Bridge)

Il bridge collega il simulatore Unreal (in esecuzione su Windows) all'ambiente ROS 2 (su WSL). 

1. Assicurati che il simulatore FSDS sia aperto e avviato su Windows.
2. In un terminale di WSL, recupera automaticamente l'IP di Windows e avvia il bridge:

```bash
source /opt/ros/jazzy/setup.bash
cd /mnt/windows_path_to/Formula-Student-Driverless-Simulator/ros2
source install/setup.bash

# Ottiene l'IP di Windows (host) da WSL
WINDOWS_IP=$(ip route | grep default | awk '{print $3}')

# Avvia il bridge interfacciato con Windows
ros2 launch fsds_ros2_bridge fsds_ros2_bridge.launch.py host:=$WINDOWS_IP
```

---

## 3. Avvio dei Nodi di Guida Autonoma

Apri tre terminali separati di WSL, esegui il source e lancia i nodi in quest'ordine:

### Terminale 1: Calcolo della Centerline
Calcola in tempo reale la centerline della pista a partire dai coni della pista visualizzati.
```bash
source /opt/ros/jazzy/setup.bash
cd /mnt/windows_path_to/Formula-Student-Driverless-Simulator/ros2
source install/setup.bash
ros2 run driverless centerline_node
```

### Terminale 2: Pianificatore di percorso RRT*
Pianifica la traiettoria ottimale aggirando i coni ed evitando le collisioni.
```bash
source /opt/ros/jazzy/setup.bash
cd /mnt/windows_path_to/Formula-Student-Driverless-Simulator/ros2
source install/setup.bash
ros2 run driverless rrt_node
```

### Terminale 3: Regolatore Pure Pursuit
Legge la traiettoria calcolata e invia i comandi di accelerazione, freno e sterzata al simulatore FSDS.
```bash
source /opt/ros/jazzy/setup.bash
cd /mnt/windows_path_to/Formula-Student-Driverless-Simulator/ros2
source install/setup.bash
ros2 run driverless pure_pursuit
```

---

## 5. Visualizzazione su RViz2

Per avviare RViz2 per il monitoraggio 3D (coni rilevati, traiettoria verde, albero di ricerca RRT*, centerline rossa e modello 3D dell'auto):

```bash
source /opt/ros/jazzy/setup.bash
rviz2
```

### Configurazione consigliata su RViz2:
* **Fixed Frame:** Impostalo scrivendo manualmente **`fsds/map`** (nella parte alta del pannello sinistro).
* **Views -> Target Frame:** Impostalo a **`fsds/FSCar`** (così la telecamera inseguirà automaticamente l'auto).
* **Views -> Type:** Impostalo a **`Orbit (rviz)`**.
* **Display Modello Auto (RobotModel):** Clicca su *Add -> By Display Type -> RobotModel*. Il modello 3D reale comparirà automaticamente ancorato alla vettura.
* **Display Traiettoria (Path):** Aggiungi un display cliccando su *Add -> By Topic -> /planning/trajectory (Path)*.
* **Display Marker (MarkerArray):**
  * Aggiungi `/planning/viz` per vedere l'albero di ricerca RRT* e la linea di traguardo locale.
  * Aggiungi `/viz/cones` per vedere i cilindri colorati che rappresentano i coni blu/gialli.
  * Aggiungi `/debug/centerline` per verificare la mesh di Delaunay e la centerline di riferimento.

### Risoluzione problemi telecamera o pannelli buggati in RViz2:
Se la telecamera 3D smette di rispondere o se il pannello delle visualizzazioni (*Views*) si scollega diventando non cliccabile:
* **Per resettare il layout:** Vai nel menu in alto di RViz su **Panels** -> **Reset**.
* **Per forzare un avvio pulito privo di vecchie configurazioni:**
  ```bash
  rviz2 -d ""
  ```

---

## 6. Architettura e Dettagli Implementativi del Path Planner RRT*

Il pacchetto `driverless` integra un pianificatore di percorso basato sull'algoritmo **Kinematic RRT*** (Rapidly-exploring Random Tree Star), ottimizzato per le competizioni Formula Student Driverless. 

L'implementazione è strutturata in due moduli principali:
1. **Nucleo Algoritmico (Core):** Implementato in [rrt_star.py](file:///C:/PROJECT/Formula-Student-Driverless-Simulator-URC/ros2/src/driverless/driverless/path_planning/rrt_star.py).
2. **Wrapper ROS 2:** Implementato in [rrt_node.py](file:///C:/PROJECT/Formula-Student-Driverless-Simulator-URC/ros2/src/driverless/driverless/path_planning/rrt_node.py).

### 6.1 Algoritmo RRT* Cinematico (`rrt_star.py`)

A differenza del classico RRT, questa versione tiene conto dei vincoli dinamici e geometrici della vettura:

* **Modello Cinematico della Vettura (Bicicletta):** Durante l'espansione dell'albero (funzione `_steer`), una Bezier quintica raccorda le pose dei nodi con curvatura continua e nulla agli estremi. La curvatura viene convertita nell'angolo di sterzo del modello a bicicletta e i raccordi oltre `max_steering_angle` vengono scartati, senza saturazioni brusche.
* **Campionamento Guidato da Centerline:** Se disponibile una linea di mezzeria (centerline), il campionamento non avviene in modo uniforme in tutto lo spazio, ma viene circoscritto entro un raggio predefinito (`sample_radius_centerline`) attorno a punti casuali della centerline. Questo accelera notevolmente la convergenza e mantiene l'albero nella carreggiata.
* **Funzione di Costo Custom:** Il costo di ogni nodo (`_calc_new_cost`) include:
  * Lunghezza geometrica reale del percorso (somma dei segmenti integrati).
  * Penale sulla curvatura (penalizza bruschi cambi di heading `theta`) per favorire traiettorie più rettilinee e fluide.
* **Riconnessione Sicura (Rewiring):**
  * La fase di scelta del padre e di ricollegamento verifica la fattibilità cinematica reale della traiettoria e l'assenza di collisioni (tramite `CollisionChecker`).
  * È implementato un controllo dei cicli per evitare loop infiniti nell'albero.
  * La propagazione dei costi migliorati ai nodi figli avviene in modo iterativo tramite **BFS** (Breadth-First Search) per prevenire il superamento del limite di ricorsione di Python.
* **Smoothing con Spline Cubica:** Il percorso finale estratto può essere addolcito tramite una spline cubica parametrizzata (`scipy.interpolate.splprep` e `splev`), che distribuisce i waypoint a distanza d'arco costante definita dal passo del pianificatore.

### 6.2 Nodo Wrapper ROS 2 (`rrt_node.py`)

Il nodo gestisce gli input sensoriali e la pubblicazione della traiettoria:

* **Trigger di Ripianificazione Intelligente:** Per ottimizzare l'uso della CPU, il pianificatore non gira costantemente a 10 Hz se la mappa dei coni è statica. Ricalcola il percorso solo se:
  * Non è ancora presente alcuna traiettoria.
  * Vengono rilevati almeno `NEW_CONE_THRESHOLD` (default: 2) nuovi coni blu **E** 2 nuovi coni gialli rispetto all'ultima pianificazione.
* **Stitching & Continuità Temporale:** Quando si ripianifica, per garantire una transizione fluida ed evitare strappi nei comandi di sterzata dell'auto, l'algoritmo non fa partire l'RRT* dalla posizione istantanea dell'auto, ma dal **4° punto dalla fine del percorso precedentemente pianificato** (se disponibile).
* **Trimming dei Waypoint Superati:** Ad ogni ciclo a 10 Hz, i waypoint che si trovano a più di 3 metri dietro la vettura (rispetto al frame locale) vengono rimossi dal percorso pubblicato.
* **Pubblicazione dei Marker per RViz:** Il nodo pubblica marker dettagliati su `/planning/viz` per scopi di debug:
  * Rami dell'albero RRT* (linee grigie) e nodi (punti celesti).
  * Punti di campionamento generati casualmente (punti arancioni).
  * Traiettoria finale (linea verde e sfere ciano per i waypoint).
  * Linea di traguardo locale (linea fucsia).
  * Coni visti all'interno dell'orizzonte locale (cubi colorati).

### 6.3 Parametri del Nodo

I seguenti parametri sono configurabili all'avvio del nodo (es. tramite file di configurazione o riga di comando):

| Parametro | Tipo | Default | Descrizione |
| :--- | :--- | :--- | :--- |
| `collision_strategy` | string | `'radial'` | Strategia per il controllo collisioni (es. `'radial'`). |
| `max_steering_angle` | float | `math.radians(24)` | Angolo massimo di sterzata delle ruote in radianti. |
| `wheelbase` | float | `1.58` | Passo della vettura in metri. |
| `step_size` | float | `1.0` | Lunghezza del passo di espansione dell'albero RRT* in metri. |
| `sample_radius_centerline` | float | `1.5` | Raggio di campionamento intorno alla centerline. |
| `max_iter` | int | `500` | Numero massimo di iterazioni per ogni ciclo di pianificazione. |
| `centerline_topic` | string | `'/track/centerline'` | Topic ROS 2 da cui ricevere la centerline. |
| `cones_topic` | string | `'/fsds/testing_only/track'` | Topic ROS 2 da cui ricevere la mappa dei coni. |
