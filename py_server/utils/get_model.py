import os
from ultralytics import YOLO

os.makedirs("../model", exist_ok=True)

model = YOLO("../model/yolo26n.pt")
model.export(format="onnx", dynamic=True)
