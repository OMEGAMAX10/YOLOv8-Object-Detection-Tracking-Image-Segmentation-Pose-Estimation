# YOLOv8 Image Segmentation, Object Detection and Tracking
YOLOv8 object detection, tracking and image segmentation using Ultralytics API (for detection and segmentation), as well as DeepSORT (for tracking) in Python.
The results of the detection are extracted in JSON format and prepared for further processing.
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
![image](https://user-images.githubusercontent.com/48774025/221048940-748600fb-f4c4-4d43-9aaa-43e9ba278ad5.png)
In this tab, the user can upload an image to be processed using the model mentioned above. The resulting image, with the bounding boxes and segmentation masks, is then beautifully displayed in this tab after the processing is finished.

### b. Video Processing Tab

In this tab, the user can upload a MP4 video to be processed. The resulting video, with the bounding boxes, segmentation masks and tracks of the identified objects, is then beautifully displayed in this tab after the processing is finished. In addition, the video and the JSON file with the objects detected is saved locally in a folder with the name of the initial video within the local folder "output_videos/" for further analysis.

### c. Live Stream Tab
![image](https://user-images.githubusercontent.com/48774025/221050282-673649d3-6cc2-4bab-b77a-4a738d0a325c.png)
In this tab, the user can add the URL for a live video stream, which can be either a integer (0, 1, 2 etc.) for a physically connected camera, like USB or built-in webcams, or a RTSP stream from a remotely connected camera. After starting the processing of the stream, the resulting frames, with the bounding boxes, segmentation masks and tracks of the identified objects, are displayed in real time. The processing of the stream can be started and stopped at any time using the two buttons "Start Live Stream Processing" and "Stop Live Stream Processing".
