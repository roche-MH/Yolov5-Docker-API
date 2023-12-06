from typing import List
import cv2
from src.utils.convert_json import results_to_json

class Predictor:
    def __init__(self, image, model):
        self.size = (640,640)
        self.resize_image = None
        self.image = image
        self.model = model

    def image_preprocess(self):
        self.resize_image = cv2.resize(self.image,self.size)

    def object_detect(self) -> List:
        self.image_preprocess()
        out = self.model(self.resize_image, size=self.size[0])

        json_result = results_to_json(out, self.model)
        return json_result
