# Code adapted from Tensorflow Object Detection Framework
# https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
# Tensorflow Object Detection Detector

import numpy as np
import tensorflow as tf
import cv2
import time
import math


class DetectorAPI:
    def __init__(self, path_to_ckpt, threshold):
        self.people = []
        self.selected_person = None

        self.threshold = threshold
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.default_graph = self.detection_graph.as_default()
        self.sess = tf.Session(graph=self.detection_graph)

        self.image_tensor = self.detection_graph.get_tensor_by_name(
            'image_tensor:0')
        self.detection_boxes = self.detection_graph.get_tensor_by_name(
            'detection_boxes:0')
        self.detection_scores = self.detection_graph.get_tensor_by_name(
            'detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name(
            'detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name(
            'num_detections:0')

    def track_people(self, new_people):
        # loop new people
            # check if centroid of new person is within 50px of old person
            # if yes, replace. mark person as found
            # if no, add to end of list as found
        # loop people
            # if not found, decrement life by 1 and leave coords alone

        for new_person in new_people:
            x, y = new_person.centroid
            matches = [person for person in self.people if math.hypot(
                person.x - x, person.y, y)]
            print(matches)
            if not matches:
                self.people.add(new_person)

        return

    def processFrame(self, image):
        image_np_expanded = np.expand_dims(image, axis=0)

        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores,
                self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})

        im_height, im_width, _ = image.shape

        scores = scores[0].tolist()
        classes = [int(x) for x in classes[0].tolist()]
        detections = []

        for i in range(boxes.shape[1]):
            pt1 = boxes[0, i, 0]
            pt2 = boxes[0, i, 1]
            pt3 = boxes[0, i, 2]
            pt4 = boxes[0, i, 3]

            pt1scaled = pt1 * im_height
            pt2scaled = pt2 * im_width
            pt3scaled = pt3 * im_height
            pt4scaled = pt4 * im_width

            centroid = (np.mean([pt2, pt4]), np.mean([pt1, pt3]))

            if (scores[i] < self.threshold or classes[i] != 1):
                continue

            detection = {
                "image_scaled_box": [int(x) for x in (pt2scaled, pt1scaled, pt4scaled, pt3scaled)],
                "centroid": (int(centroid[0] * 100), int(centroid[1] * 100)),
                "image_scaled_centroid": (int(centroid[0] * im_width), int(centroid[1] * im_height))
            }

            detections.append(detection)

        # TODO select a random person on timer
        # TODO return coords of the selected person

        return detections

    def close(self):
        self.sess.close()
        self.default_graph.close()
