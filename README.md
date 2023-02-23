# YOLOv8 Image Segmentation, Object Detection and Tracking
YOLOv8 object detection, tracking and image segmentation using Ultralytics API (for detection and segmentation), as well as DeepSORT (for tracking) in Python.
The results of the detection are extracted in json format and prepared for further processing.
For an enhanced user experience, the app interface is built using "streamlit" module from Python. The app can also be deployed in a Docker container.

## 1. App Deployment
The app can be deployed and run using the following commands:
```
sudo docker build --tag yolov8-docker .
sudo docker run -d --gpus all --name yolov8-docker -p 8501:8501 yolov8-docker
```

## 2. App Usage
The web app consists of three tabs in which the user can do image, video or live stream processing using the model [YOLOv8 developed by Ultralytics](https://github.com/ultralytics/ultralytics). The application is composed of three tabs as follows:

### a. Image Processing Tab
![image](https://user-images.githubusercontent.com/48774025/221047930-d7603ba5-a8bf-43d8-b508-377c14eff2c0.png)
In this tab, the user can send an image to the app to be processed using the model mentioned above. The resulting image, with the bounding boxes and segmentation masks, is then beautifully displayed in this tab after the processing is finished.

### b. Video Processing Tab

### c. Live Stream Tab
