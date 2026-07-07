import math


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