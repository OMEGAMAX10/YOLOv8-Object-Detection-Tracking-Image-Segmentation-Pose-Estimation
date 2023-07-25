# YOLOv8 Object Detection, Tracking, Image Segmentation and Pose Estimation
YOLOv8 object detection, tracking, image segmentation and pose estimation using Ultralytics API (for detection, pose estimation and segmentation), as well as DeepSORT (for tracking) in Python.
The results of the detection are extracted in JSON format and prepared for further processing.
For an enhanced user experience, the app interface is built using "streamlit" module from Python. The app can also be deployed in a Docker container.
This app is also available as a public container on [Docker Hub](https://hub.docker.com/r/bmarghescu/yolov8-docker).

## 1. App Deployment
The app can be deployed using Docker Compose using the following command:
```
docker-compose -f yolov8-docker.yml up -d
```
If you want to deploy the app using the official container from Docker Hub, use the following commands:
```
docker pull bmarghescu/yolov8-docker
docker run -d --gpus all --name yolov8-docker -p 80:8501 bmarghescu/yolov8-docker:latest
```

## 2. App Usage
The web app consists of four tabs in which the user can do image, video or live stream processing using the model [YOLOv8 developed by Ultralytics](https://github.com/ultralytics/ultralytics), as well as upload a custom YOLOv8 model. The application is composed of four tabs as follows:

### a. Image Processing Tab
![image](https://user-images.githubusercontent.com/48774025/221048940-748600fb-f4c4-4d43-9aaa-43e9ba278ad5.png)
In this tab, the user can upload an image to be processed using the model mentioned above. The resulting image, with the bounding boxes and segmentation masks, is then beautifully displayed in this tab after the processing is finished.

### b. Video Processing Tab
![image](https://user-images.githubusercontent.com/48774025/221053190-57d17253-d9c9-4a7b-8616-85f565a40dd4.png)
In this tab, the user can upload a MP4 video to be processed. The resulting video, with the bounding boxes, segmentation masks and tracks of the identified objects, is then beautifully displayed in this tab after the processing is finished. The video can also be viewed in full screen mode for better visualization of the result. In addition, the video and the JSON file with the objects detected is saved locally in a folder with the name of the initial video within the local folder **"output_videos/"** for further analysis.

### c. Live Stream Tab
![image](https://user-images.githubusercontent.com/48774025/221050282-673649d3-6cc2-4bab-b77a-4a738d0a325c.png)
In this tab, the user can add the URL for a live video stream, which can be either a integer (0, 1, 2 etc.) for a physically connected camera, like USB or built-in webcams, or a RTSP stream from a remotely connected camera. After starting the processing of the stream, the resulting frames, with the bounding boxes, segmentation masks and tracks of the identified objects, are displayed in real time. The processing of the stream can be started and stopped at any time using the two buttons "Start Live Stream Processing" and "Stop Live Stream Processing".

### d. Custom YOLOv8 Model Upload Tab
![image](https://github.com/BogdanMarghescu/YOLOv8-Image-Segmentation-Object-Detection-and-Tracking/assets/48774025/651192b1-ba55-4e8d-ab2f-d0810309f9ab)
In this tab, the user can upload a custom YOLOv8 model, trained on a custom dataset. The model is stored in a folder named "models/" and added to "model_list.txt" for future use. After it was added, the model can then be selected in the dropdown menu at the beginning of the page, and it is usually found at the end of the list of the predefined models. The only eligible type of model is the Pytorch (".pt") one.
