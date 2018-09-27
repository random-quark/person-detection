# Code adapted from Tensorflow Object Detection Framework
# https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
# Tensorflow Object Detection Detector

import numpy as np
import tensorflow as tf
import cv2
import time
import math
import atexit
import pickle
import os

debug = True

global data

# def exit_handler():
#     with open('data_cache.pickle', 'wb') as file:
#         pickle.dump(data, file)


# atexit.register(exit_handler)


def load():
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, 'data_cache.pickle')
    with open(filename, 'rb') as file:
        return pickle.load(file)


class DetectorAPI:
    def mockDetection(self, frameIndex):
        return self.data[frameIndex]

    def __init__(self, relative_path_to_ckpt, threshold, allowed_movement_per_frame, allowed_tracking_loss_frames):
        self.data = load()  # get dummy data

        # FIXME: need better way of supplying infinite unique names
        self.names_source = ["Tom", "Simon", "Leslie", "John",
                             "Peter", "Ruby", "Sioban", "Ella", "Jane"]
        self.names = self.names_source.copy()

        self.people = {}
        self.allowed_movement_per_frame = allowed_movement_per_frame
        self.allowed_tracking_loss_frames = allowed_tracking_loss_frames

        self.threshold = threshold

        if not debug:
            self.detection_graph = tf.Graph()
            with self.detection_graph.as_default():
                od_graph_def = tf.GraphDef()
                dirname = os.path.dirname(__file__)
                path_to_ckpt = os.path.join(dirname, relative_path_to_ckpt)
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

    def track_people(self, detections):
        # FIXME break up into smaller functions
        for detection in detections:
            x, y = detection["centroid"]

            matches = [key for key, person in self.people.items() if math.hypot(
                person["centroid"][0] - x, person["centroid"][1] - y) < self.allowed_movement_per_frame]
            distances = [math.hypot(self.people[name]["centroid"][0] - x,
                                    self.people[name]["centroid"][0] - y) for name in matches]

            matches = [match for _, match in sorted(
                zip(distances, matches))]  # sort by closest existing match

            if matches:
                self.people[matches[0]]["centroid"] = detection["centroid"]
                self.people[matches[0]
                            ]["image_scaled_box"] = detection["image_scaled_box"]
                self.people[matches[0]
                            ]["image_scaled_centroid"] = detection["image_scaled_centroid"]
                self.people[matches[0]
                            ]["health"] = self.allowed_tracking_loss_frames
            else:
                detection["health"] = self.allowed_tracking_loss_frames
                # FIXME: better solution for infinite source of names. counter?
                if not self.names:
                    self.names = self.names_source.copy()
                self.people[self.names.pop()] = detection.copy()

        for person in self.people.values():
            person["health"] -= 1
        remove = [k for k, person in self.people.items()
                  if person["health"] <= 1]
        for k in remove:
            del self.people[k]

        return

    def machine_learning_data(self, image):
        image_np_expanded = np.expand_dims(image, axis=0)
        (boxes, scores, classes) = self.sess.run(
            [self.detection_boxes, self.detection_scores,
                self.detection_classes],
            feed_dict={self.image_tensor: image_np_expanded})
        data.append((boxes, scores, classes))  # collect data
        return (boxes, scores, classes)

    def cached_data(self, frame):
        (boxes, scores, classes, _) = self.mockDetection(
            frame)  # fast for testing
        return (boxes, scores, classes)

    def processFrame(self, image, frame=0):
        (boxes, scores, classes) = self.cached_data(
            frame) if debug else self.machine_learning_data(image)

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
                "centroid": (int(centroid[0] * 100), int(centroid[1] * 100)),
                "image_scaled_box": [int(x) for x in (pt2scaled, pt1scaled, pt4scaled, pt3scaled)],
                "image_scaled_centroid": (int(centroid[0] * im_width), int(centroid[1] * im_height))
            }

            detections.append(detection)

        self.track_people(detections)

        return self.people.copy()

    def close(self):
        self.sess.close()
        self.default_graph.close()
