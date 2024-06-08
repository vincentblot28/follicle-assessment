from ultralytics import YOLO

model = YOLO('yolov8l.pt')  # load a pretrained model (recommended for training)
results = model.train(
    data='data/data_yolo.yml', epochs=100, imgsz=640
)
