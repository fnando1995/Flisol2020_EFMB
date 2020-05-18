from tracking.sort import Sort

import cv2
import numpy as np
import json

def put_tracked_in_frame(tracked,frame):
    for trk in tracked:
        # bbox drawing
        bbox = trk.get_state()[0]
        startX, startY,endX, endY = np.array(bbox[:4]).astype(int)
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0,255,0), 2)
        # centroid circle drawing
        circle = trk.get_point_of_interest()
        circle = (int(circle[0]),int(circle[1]))
        # id text drawing
        texto = str(trk.get_id())
        cv2.putText(frame, texto, circle, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
    return frame

class Tracker(object):
    def __init__(self,THRESH,FPS = 10):
        self.algorithm              =   Sort(THRESH)
        self.tracked_detections     =   []
        self.video_fps              =   FPS
        self.total_counts           =   0


    def reset_tracks(self):
        self.tracked_detections =   []


    def track_dets(self,dets,frame):
        """
        param dets: [x1,y1,x2,y2,class],conf]
        :param frame:
        :return:
        """
        new_tracked_detections,erased_trks = self.algorithm.update(dets,self.tracked_detections)
        # Algoritmo ultra-mega-simplista-jamas-recomendado-para-conteo.
        for erased in erased_trks:
            if erased.get_age()/self.video_fps > 2.0:
                self.total_counts += 1
        self.tracked_detections = new_tracked_detections
        return put_tracked_in_frame(self.tracked_detections,frame),self.total_counts





