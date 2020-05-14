import cv2
from modelos.ssd_mobilenet_v2_coco.vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
import os
BASE_DIR = "/".join(os.path.realpath(__file__).split("/")[:-2])



class Network():
    model_path = BASE_DIR + "/modelos/ssd_mobilenet_v2_coco/models/mobilenet-v1-ssd-mp-0_675.pth"
    label_path = BASE_DIR + "/modelos/ssd_mobilenet_v2_coco/models/voc-model-labels.txt"


    def __init__(self):
        self.class_names = [name.strip() for name in open(self.label_path).readlines()]
        self.num_classes = len(self.class_names)
        self.net = create_mobilenetv1_ssd(len(self.class_names), is_test=True)
        self.net.load(self.model_path)
        self.predictor = create_mobilenetv1_ssd_predictor(self.net, candidate_size=200)

    def predict(self,image,prob,return_image=True,top_k=-1):
        boxes, labels, probs = self.predictor.predict(image, top_k, prob) #top_k: if -1 keeps all
        detections = []
        for b,l,p in zip(boxes,labels,probs):
            detections.append([ int(i) for i in list(b)]+[int(l)]+[round(float(p),4)])

        if return_image:
            return self.put_data_in_frame(detections,image)
        else:
            return detections

    def put_data_in_frame(self,detections,orig_image):
        for det in detections:
            box = det[:4]
            label = f"{self.class_names[det[4]]}: {det[5]:.2f}"
            cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)
            cv2.putText(orig_image, label,
                        (box[0]+20, box[1]+40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,  # font scale
                        (255, 0, 255),
                        2)  # line type
        return detections,orig_image


