FROM nvcr.io/nvidia/pytorch:21.10-py3
ENV PYTHONPATH=/usr/lib/python3.9/site-packages
RUN apt-get update \
    && apt-get install -y --no-install-recommends apt-utils libgl1 libglib2.0-0 \
    python3-pip \
    && apt-get install psmisc \
    && apt-get clean \
    && apt-get autoremove
RUN pip3 install --upgrade pip
RUN mkdir /object-detection-yolov5
WORKDIR /object-detection-yolov5
COPY . /object-detection-yolov5
RUN pip install -r requirements.txt
RUN pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org python-multipart
EXPOSE 8000
CMD ["python", "apis/app.py"]
