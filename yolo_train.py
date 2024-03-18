from ultralytics import YOLO

model = YOLO('./original_yolo_models/yolov8n.pt')

results = model.train(data='data.yaml', epochs=20, imgsz=640)
