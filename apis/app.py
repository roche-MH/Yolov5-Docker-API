import os
import logging
from io import BytesIO
from typing import List, Optional
from warnings import filterwarnings, simplefilter #TODO ? 
import ssl
import torch
import uvicorn
from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel
import sys
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '../')))
from src.predictor import Predictor


#ssl 에러 발생시
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context
import cv2
import base64
import io
import numpy as np

filterwarnings("ignore")
simplefilter(action="ignore", category=FutureWarning)

if not os.path.exists('../logs'):
    os.mkdir("../logs")

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.StreamHandler()
file_handler = logging.FileHandler("../logs/api.log")
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s : %(levelname)s : %(name)s : %(message)s")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

app: FastAPI = FastAPI()
try:
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/best.pt')
except:
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=True)

def stringToRGB(base64_string):
    imgdata = base64.b64decode(base64_string)
    dataBytesIO = io.BytesIO(imgdata)
    image = Image.open(dataBytesIO)
    return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)


class Item(BaseModel):
    cam_index: Optional[str] = None
    image: Optional[str] = None
    date: Optional[str] = None

@app.post("/detector/")
async def image_detect(item: Item):
    try:
        json_result: List = []
        img = stringToRGB(item.image)
        ob: ObjectDetector = Predictor(img, model)
        json_results = ob.object_detect()

        logger.info(["object detection results", json_result])

        return JSONResponse(
            {
                        "data": json_results,
                        "message": "object detected successfully",
                        "errors": None,
                        "status": 200,
            },
            status_code=200,
        )
    except Exception as error:
        logger.error(["process failed", error])
        return JSONResponse(
            {
                    "message": "object detection failed",
                    "errors": "error",
                    "status": 400,
            },
            status_code=400,
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, debug=True)
