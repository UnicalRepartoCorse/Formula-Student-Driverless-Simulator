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

def normalize_angle(angle):
    """
    Normalize an angle to the range [-π, π).

    This function takes an input angle and adjusts it to be within
    the standard range of -π (inclusive) to π (exclusive). It ensures
    that angles are represented in a consistent format, which can be
    useful for computations in geometry, physics, and engineering.

    :param angle: The angle in radians to normalize.
    :type angle: float
    :return: The normalized angle in radians within the range [-π, π).
    :rtype: float
    """
    return (angle + math.pi) % (2*math.pi) - math.pi

def dist_sq(p,q):
    return (p[0] - q[0])**2 + (p[1] - q[1])**2