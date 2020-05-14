from inferences.non_optimized import Network
from tracking.tracker import Tracker
import cv2
import os
BASE_DIR = "/".join(os.path.realpath(__file__).split("/")[:-1])

trk_thresh = 0.4
det_thresh = 0.5

net   = Network()
video = BASE_DIR + "/video/Pedestrian_Detect_2_1_1.mp4"
video = cv2.VideoCapture(video)
FPS = video.get(cv2.CAP_PROP_FPS)
Trk = Tracker(trk_thresh,FPS)

while True:
    _,frame = video.read()
    if not _:
        break
    detections = net.predict(frame,det_thresh,False,10)
    image, conteo = Trk.track_dets(detections,frame)
    print("Personas ingresadas:",conteo)
    cv2.imshow("win",image)
    cv2.waitKey(1)

cv2.destroyAllWindows()