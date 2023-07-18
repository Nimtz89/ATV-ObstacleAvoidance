import cv2
import os
from time import sleep
from msrest.authentication import ApiKeyCredentials
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient

predictionKey = os.environ.get("predictionKey")
customvisionendpoint = os.environ.get("customvisionendpoint")
project_id = os.environ.get("project_id")
itterationname = os.environ.get("itterationname")

# Custom vision credentials
apikeycredentials = ApiKeyCredentials(in_headers={"Prediction-key": predictionKey})
cvpredictor= CustomVisionPredictionClient(customvisionendpoint, apikeycredentials)

camera = cv2.VideoCapture(0)
if not camera.isOpened():
 print("Cannot open camera")
 exit()

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

try:
    while True:
        ret, image = camera.read()
        cv2.imwrite("picture_captured.png", image)
        with open("picture_captured.png", mode="rb") as picture_captured:
            results = cvpredictor.detect_image(project_id, itterationname, picture_captured)

        for prediction in results.predictions:
           print("prediciton probability: ", prediction.probability)
           sleep(3)
           

except KeyboardInterrupt:
    camera.release()
