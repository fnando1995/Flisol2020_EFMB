from __future__ import print_function
import numpy as np
from filterpy.kalman import KalmanFilter


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h  # scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if (score == None):
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))


class KalmanBoxTracker(object):
    """
    This class represents the internel state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, bbox, init_time, point_of_interest="centroid"):
        """
        Initialises a tracker using initial bounding box.
        """
        # define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array(
            [[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array(
            [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])
        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        self.kf.x[:4] = convert_bbox_to_z(bbox[:4])  # convert_bbox_to_z(bbox)


        self.time_since_update = 0
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.objConfidence = bbox[4]
        self.objclass = bbox[5]
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1

        self.point_of_interest = point_of_interest
        self.init_time = init_time
        self.end_time = None


    """
    Updates the state vector with observed bbox.
    """

    def update(self, bbox):

        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    """
    Advances the state vector and returns the predicted bounding box estimate.
    """

    def predict(self):
        """
        Realiza la prediccion de la siguiente posicion del filtro.
        """
        if ((self.kf.x[6] + self.kf.x[2]) <= 0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if (self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        retorna la caja de deteccion estimada.
        """
        return convert_x_to_bbox(self.kf.x)

    def get_point_of_interest(self):
        """
        Retorna el punto de interes a medir
        """
        if self.point_of_interest == "centroid":
            return (round(float(self.kf.x[0][0]), 2), round(float(self.kf.x[1][0]), 2))
        elif self.point_of_interest == "botmid":
            x1, y1, x2, y2 = convert_x_to_bbox(self.kf.x)[0]
            x = (x2 + x1) / 2
            y = y2
            return (round(float(x), 2), round(float(y), 2))
        elif self.point_of_interest == "topmid":
            x1, y1, x2, y2 = convert_x_to_bbox(self.kf.x)[0]
            x = (x2 + x1) / 2
            y = y1
            return (round(float(x), 2), round(float(y), 2))
        else:
            print("point of interest not devised")
            exit()

    def get_id(self):
        return self.id

    def get_age(self):
        return self.age

    def set_end_time(self,end_time):
        self.end_time = end_time

    def get_time_in_seconds(self):
        return round((self.end_time - self.init_time).total_seconds(),4)