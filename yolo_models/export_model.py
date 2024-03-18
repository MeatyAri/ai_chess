from ultralytics import YOLO


model = YOLO('yolo_models/yolov8n_chess_board.pt')

model.export(format='engine', imgsz=640)
