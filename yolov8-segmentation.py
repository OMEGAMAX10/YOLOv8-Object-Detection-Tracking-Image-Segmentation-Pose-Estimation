import os
import cv2
import json
from ultralytics import YOLO
from distinctipy import distinctipy

CLASS_LIST = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
    'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
    'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
    'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
    'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
CLASS_COLORS = distinctipy.get_colors(len(CLASS_LIST))
CLASS_COLORS = [tuple([int(x * 256) for x in color]) for color in CLASS_COLORS]


def object_segmentation(image, model, temp_filename):
    ALPHA = 0.5
    results = model.predict(source=temp_filename, conf=0.25, return_outputs=True)
    result_list_json = []
    for result in results:
        if len(result['segment']) != len(result['det']):
            raise Exception('Number of segmentation and detection results do not match')
        image_mask = image.copy()
        for idx in range(len(result['segment'])):
            segment = result['segment'][idx].astype(int)
            detection = result['det'][idx]
            result_json = {
                'class': CLASS_LIST[int(detection[5])],
                'confidence': float(detection[4]),
                'bbox': {
                    'x_min': int(detection[0]),
                    'y_min': int(detection[1]),
                    'x_max': int(detection[2]),
                    'y_max': int(detection[3]),
                },
                'segmentation': segment.tolist(),
            }
            result_list_json.append(result_json)
            class_color = CLASS_COLORS[int(detection[5])]
            class_color = (class_color[0], class_color[1], class_color[2], 128)
            cv2.fillPoly(image_mask, [segment], class_color)
        image = cv2.addWeighted(image, ALPHA, image_mask, 1 - ALPHA, 0)
        for idx in range(len(result['det'])):
            detection = result['det'][idx]
            text = f"{CLASS_LIST[int(detection[5])]}: {detection[4] * 100:.2f}%"
            class_color = CLASS_COLORS[int(detection[5])]
            text_color = tuple([256 - x for x in class_color])
            cv2.rectangle(image, (int(detection[0]), int(detection[1])), (int(detection[2]), int(detection[3])), class_color, 2)
            (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image, (int(detection[0]), int(detection[1])), (int(detection[0]) + text_width, int(detection[1]) - text_height - baseline), class_color, -1)
            cv2.putText(image, text, (int(detection[0]), int(detection[1]) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
    os.remove(temp_filename)
    return image, result_list_json


# Model initialization
model = YOLO('yolov8s-seg.pt')

# YOLOv8S-SEGMENTATION - IMAGE
IMAGE_PATH = "images/family-and-dog.jpg"
image = cv2.imread(IMAGE_PATH)
cv2.imwrite('temp.jpg', image)
image, result_list_json = object_segmentation(image, model, 'temp.jpg')
# print(json.dumps(result_list_json, indent=2))
cv2.imshow(f"Image - Source: {IMAGE_PATH}", image)
cv2.waitKey(0)

# YOLOv8S-SEGMENTATION - REAL-TIME CAMERA
CAM_ID = 0
cam = cv2.VideoCapture(CAM_ID)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
while True:
    _, image = cam.read()
    cv2.imwrite('temp.png', image)
    image, result_list_json = object_segmentation(image, model, temp_filename='temp.png')
    # print(json.dumps(result_list_json, indent=2))
    cv2.imshow(f"Webcam - Source: {CAM_ID}", image)
    if cv2.waitKey(1) == 27:  # ESC
        break
cam.release()
cv2.destroyAllWindows()
