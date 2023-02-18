import os
import cv2
import json
from ultralytics import YOLO
from ultralytics.yolo.engine.results import Results


def result_to_json(result: Results):
    result_list_json = [
        {
            'class': result.names[int(result.boxes.cls[idx])],
            'confidence': float(result.boxes.conf[idx]),
            'bbox': {
                'x_min': int(result.boxes.boxes[idx][0]),
                'y_min': int(result.boxes.boxes[idx][1]),
                'x_max': int(result.boxes.boxes[idx][2]),
                'y_max': int(result.boxes.boxes[idx][3]),
            },
            'mask': cv2.resize(result.masks.data[idx].cpu().numpy(), (result.orig_shape[1], result.orig_shape[0])).tolist(),
            'segments': result.masks.segments[idx].tolist(),
        } for idx in range(len(result))
    ]
    return result_list_json


def object_segmentation(model, temp_img_filename):
    results = model.predict(temp_img_filename)
    os.remove(temp_img_filename)
    result_list_json = result_to_json(results[0])
    result = results[0].cpu()
    result.names = results[0].names
    image = result.visualize()
    return image, result_list_json


# Model initialization
model = YOLO('yolov8s-seg.pt')

# YOLOv8S-SEGMENTATION - IMAGE
IMAGE_PATH = "images/family-and-dog.jpg"
image = cv2.imread(IMAGE_PATH)
cv2.imwrite('temp.jpg', image)
image, result_list_json = object_segmentation(model, temp_img_filename='temp.jpg')
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
    image, result_list_json = object_segmentation(model, temp_img_filename='temp.png')
    # print(json.dumps(result_list_json, indent=2))
    cv2.imshow(f"Webcam - Source: {CAM_ID}", image)
    if cv2.waitKey(1) == 27:  # ESC
        break
cam.release()
cv2.destroyAllWindows()
