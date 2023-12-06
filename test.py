import cv2
import base64
import requests
import json

img = cv2.imread('test.jpg')
data =  {}
binary_cv = cv2.imencode('.jpg',img)
result = base64.b64encode(binary_cv[1]).decode('utf-8')
data["image"] = result
response = requests.post(url='http://10.100.0.89:8000/detector/',data=json.dumps(data))
print(response.json())

