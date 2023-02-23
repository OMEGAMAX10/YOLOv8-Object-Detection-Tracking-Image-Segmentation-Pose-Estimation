FROM nvcr.io/nvidia/pytorch:23.01-py3
RUN apt update && \
    apt install -y libsndfile1 ffmpeg libsm6 libxext6 libgl1 && \
    apt clean

WORKDIR /yolov8-docker
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

CMD [ "python3", "-m", "streamlit", "run", "yolov8-segmentation-tracking.py" ]
