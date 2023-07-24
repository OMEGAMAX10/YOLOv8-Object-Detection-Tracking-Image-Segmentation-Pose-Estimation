FROM python:3.9-slim-buster
RUN apt update && \
    apt install -y libsndfile1 ffmpeg libsm6 libxext6 libgl1 build-essential curl software-properties-common  && \
    apt clean

WORKDIR /yolov8-docker
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT [ "streamlit", "run", "yolov8-detection-tracking-segmentation-pose.py", "--server.port=8501", "--server.address=0.0.0.0" ]
