from ultralytics import YOLO


class YOLOv10Warpper:
    def __init__(self, model_path: str):
        self.model_path = model_path
        
    def detect(self, image):
        model = YOLO(self.model_path)
        results = model(image)
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0].item())
                if class_id == 0:
                    confidence = float(box.conf[0].item())
                    if confidence > 0.5:
                        x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
                        bw = float(x2 - x1)
                        bh = float(y2 - y1)
                    
                        detections.append((x1, y1, bw, bh, 0))
            
        return detections