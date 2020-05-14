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
    def __init__(self,THRESH,MatrixOPeration,AssigProblemSolver,FPS = 10):
        self.algorithm              =   Sort(THRESH,MatrixOPeration,AssigProblemSolver)
        self.tracked_detections     =   []
        self.tracks_time_thresh     =   2.0     # seconds
        self.video_fps              =   FPS
        self.tracked_frames_number  =   0   # [sum of seconds,number of tracks],
                                            # I will be using a threshold for avoiding
                                            # tends to zero with false negatives detections

    def reset_tracks(self):
        self.tracked_detections =   []


    def track_dets(self,dets,frame,client):
        new_tracked_detections,erased_trks = self.algorithm.update(dets,self.tracked_detections)
        self.analize_tracks(new_tracked_detections,erased_trks,client)
        self.tracked_detections = new_tracked_detections
        return put_tracked_in_frame(self.tracked_detections,frame)

    def analize_tracks(self,new_tracks,erased_tracks,client):
        if len(erased_tracks)!=0:
            for erased in erased_tracks:
                frames_count = erased.get_age()
                seconds = frames_count/self.video_fps
                if seconds > self.tracks_time_thresh:
                    self.tracked_frames_number += frames_count
                    average_time = self.tracked_frames_number/self.video_fps
                    client.publish("person/duration", json.dumps({"duration": average_time}))
                # e = erased.get_time_in_seconds()
                # if e > self.tracks_time_thresh:
                #     self.tracked_times[0] += e
                #     self.tracked_times[1] += 1
                #     average_time = self.tracked_times[0]#/self.tracked_times[1]
                #     client.publish("person/duration", json.dumps({"duration": average_time}))
        if len(self.tracked_detections) != new_tracks:
            total_counts = len(new_tracks)
            # print(total_counts)
            client.publish("person", json.dumps({"count": total_counts}))



