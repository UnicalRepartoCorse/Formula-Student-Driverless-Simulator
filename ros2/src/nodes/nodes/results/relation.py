import numpy as np
import ast
import matplotlib as plt

sx, sy = 0.228, 0.325
bx, by = 0.285, 0.505

max_small_area = sx*sy
max_big_area = bx*by


cx, cy, fx, fy = 392.5, 392.5, 392.5, 392.5

car_x, car_y, car_z = 1.4648437172581907e-05, -9.765624781721272e-06, 0.246235653758049
car_ox, car_oy, car_oz, car_w = 3.896256021107547e-05, -2.8252596166566946e-05, 1.1007935674101077e-09, 0.9999998807907104

cam_x, cam_y, cam_z = -0.3, 0, 0.8

cones = []
with open("all_cone_positions.txt", "r") as f:
    f.readline()
    nextline = f.readline()
    while nextline.startswith("- location:"):
        x = float(f.readline().split(":")[1].strip())
        y = float(f.readline().split(":")[1].strip())
        z = float(f.readline().split(":")[1].strip())
        color = float(f.readline().split(":")[1].strip())
        nextline = f.readline()
        if x<=0.:
            continue

        distance = np.sqrt((x-car_x)**2+(y-car_y)**2)
        cones.append((distance,x,y,color))

bboxes=[]
with open("coords.txt", "r") as f:
    lines = f.readlines()

for line in lines:
    useful = line.split("class: ")[1]
    color = useful[0]
    coords = ast.literal_eval(useful[10:].strip())
    bboxes.append((color, coords))



def camera_to_world_ENU(coords, true_area=None, true_w=None):
    x1, y1, x2, y2 = coords

    x = (x2+x1)/2.
    y = y2

    x_c, y_c, z_c = (x-cx)/fx, (y-cy)/fy, 1.

    r_x, r_y, r_z = z_c, -x_c, -y_c
    norm = np.sqrt(r_x**2 + r_y**2 + r_z**2)
    rx, ry, rz = r_x/norm, r_y/norm, r_z/norm
    
    if rz >= 0: return None
    t = -cam_z / rz

    p_w, p_h = (x2-x1), (y2-y1)
    if true_w is not None:
        t = true_w * fx / p_w
    elif true_area is not None:
        t = np.sqrt(true_area * fx * fy / (p_w*p_h))
    
    #Local coordinates relative to car center
    x_local = (t*rx)+cam_x
    y_local = (t*ry)+cam_y
    
    siny_cosp = 2 * (car_w * car_oz + car_ox * car_oy)
    cosy_cosp = 1 - 2 * (car_oy * car_oy + car_oz * car_oz)
    car_yaw = np.arctan2(siny_cosp, cosy_cosp) 

    heuristic_factor = (1+p_w/(2*fx))
    #print(heuristic_factor)

    #to global
    x_global = car_x + (x_local * np.cos(car_yaw) - y_local * np.sin(car_yaw))
    y_global = car_y + (x_local * np.sin(car_yaw) + y_local * np.cos(car_yaw))
    
    return heuristic_factor*x_global, heuristic_factor*y_global


### matching
matrix = np.array(cones)
matrix = matrix[matrix[:, 0].argsort()][1:11]


id_1 = matrix[0]
id_0 = matrix[1]
id_3 = matrix[2]
id_2 = matrix[5]
id_4 = matrix[7]
id_5 = matrix[9]

ordered = np.array([id_0, id_1, id_2, id_3, id_4, id_5])

print(ordered)

for i in range(6):
    gtx, gty, gt_color = ordered[i][1], ordered[i][2], int(ordered[i][3])
    color, bbox = int(bboxes[i][0]), bboxes[i][1]

    assert gt_color==color

    true_area = max_small_area if color<2 else max_big_area
    true_w = sx if color<2 else bx

    glob_x, glob_y = camera_to_world_ENU(bbox)
    glob_x_a, glob_y_a = camera_to_world_ENU(bbox, true_area)
    glob_x_w, glob_y_w = camera_to_world_ENU(bbox, true_w)

    diff_x, diff_y = gtx-glob_x, gty-glob_y    
    #diff_xa, diff_ya = gtx-glob_x_a, gty-glob_y_a
    #diff_xw, diff_yw = gtx-glob_x_w, gty-glob_y_w

    print(f"error on x: {diff_x}, error in y: {diff_y} for id: {i}")
    #print(f"error using area on x: {diff_xa}, error in y: {diff_ya} for id: {i}")
    #print(f"error using width on x: {diff_xw}, error in y: {diff_yw} for id: {i}")


