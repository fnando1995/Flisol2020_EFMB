from inferences.optimized import Network
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

_,frame = video.read()
net.execute_net_asyn(frame, 0)
while True:
    last_frame = frame
    _,frame = video.read()
    if not _:
        break
    net.wait(0)
    output = net.get_output(0)
    net.execute_net_asyn(frame, 0)
    image_output, detections = net.parse_outputs(output, last_frame, det_thresh)
    image, conteo = Trk.track_dets(detections,frame)
    print("Personas ingresadas:",conteo)
    cv2.imshow("win",image)
    cv2.waitKey(1)
cv2.destroyAllWindows()
