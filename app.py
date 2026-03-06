import cv2
import numpy as np
from ultralytics import YOLO
from fruit_database import fruit_database

model = YOLO("yolov8n.pt")

def detect_ripeness(crop):

    if crop is None or crop.size == 0:
        return "Unknown"

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue = np.mean(hsv[:,:,0])

    if hue < 20:
        return "Ripe"
    elif hue < 40:
        return "Mid-Ripe"
    else:
        return "Unripe"


def detect_fruit(image_path):

    img = cv2.imread(image_path)

    results = model(img)

    best_box = None
    best_conf = 0
    best_class = None

    for r in results:
        for box in r.boxes:

            conf = float(box.conf[0])
            cls = int(box.cls[0])
            name = model.names[cls].lower()

            if name in fruit_database and conf > best_conf:
                best_conf = conf
                best_box = box
                best_class = name

    if best_box is None:
        return img, None

    x1,y1,x2,y2 = map(int,best_box.xyxy[0])

    crop = img[y1:y2,x1:x2]

    ripeness = detect_ripeness(crop)

    info = fruit_database[best_class]

    cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),3)

    return img,{
        "fruit":best_class.title(),
        "ripeness":ripeness,
        "hybrid":info["hybrid"],
        "storage":info["storage_days"],
        "nutrients":info["nutrients"]
    }
