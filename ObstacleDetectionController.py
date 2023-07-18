import cv2
import os
from dotenv import load_dotenv
from time import sleep
from msrest.authentication import ApiKeyCredentials
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient

load_dotenv()

predictionKey = os.environ.get("predictionKey")
customvisionpredictionendpoint = os.environ.get("customvisionpredictionendpoint")
project_id = os.environ.get("project_id")
itterationname = os.environ.get("itterationname")


# Custom vision credentials
apikeycredentials = ApiKeyCredentials(in_headers={"Prediction-key": predictionKey})
cvpredictor= CustomVisionPredictionClient(customvisionpredictionendpoint, apikeycredentials)

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
           print("prediciton probability: ", prediction)
           sleep(3)
           

except KeyboardInterrupt:
    camera.release()
