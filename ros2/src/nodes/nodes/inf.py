import ultralytics
from . import inject
from ultralytics import YOLO
import ultralytics.nn.modules as modules
import ultralytics.nn.tasks as tasks

modules.DW = inject.DW
tasks.DW = inject.DW

modules.CDW = inject.CDW
tasks.CDW = inject.CDW

model_path_opvino = "/home/lenovo/Formula-Student-Driverless-Simulator/ros2/src/nodes/nodes/custom_b_openvino_model"
model_path = "/home/lenovo/Formula-Student-Driverless-Simulator/ros2/src/nodes/nodes/custom_b.pt"

m_quant = YOLO(model_path_opvino)
m_full = YOLO(model_path)

