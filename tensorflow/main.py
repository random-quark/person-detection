
import argparse
import numpy as np
import tensorflow as tf
import cv2
import random
import os
from detector import DetectorAPI
from pythonosc import osc_message_builder
from pythonosc import udp_client


def visualise(img, people, scores):
    im_height, im_width, _ = img.shape
    for name, person in people.items():
        box = person["image_scaled_box"]
        color = (0, 0, 255) if person.get("selected") else (0, 255, 0)
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
        cv2.circle(img, person["image_scaled_centroid"], 10, color, -1)
        textCoords = tuple(
            [pos - 10 for pos in person["image_scaled_centroid"]])
        cv2.putText(img, name, (textCoords),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color)

    cv2.putText(img, "Bounding box score: {}".format(
        scores["box_score"]), (10, im_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
    cv2.putText(img, "Number people score: {}".format(
        scores["people_score"]), (10, im_height - 50),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))

    cv2.imshow("preview", img)
    cv2.moveWindow("preview", 0, 0)
    # FIXME this probably shouldnt be here, but capture doesnt work without it
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        return


# FIXME uses the name to identify people which is eventually recycled - use a unique ID?
class PersonFinder:
    def __init__(self):
        self.track_for_frames = 50
        self.selected_person = None
        self.ttl = self.track_for_frames

    def select(self, people):
        self.selected_person = random.choice(list(people.keys()))
        self.ttl = self.track_for_frames
        return self.selected_person

    def get(self, people):
        if not self.selected_person or self.selected_person not in people or self.ttl <= 0:
            return self.select(people)
        else:
            self.ttl -= 1
            return self.selected_person


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video", default="../sample_videos/park_day_short.mov")
    parser.add_argument(
        "--model", default="./models/faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb")
    parser.add_argument("--threshold", default=0.7)
    parser.add_argument("--ip", default="192.168.1.108",
                        help="The ip of the OSC server")
    parser.add_argument("--port", type=int, default=8060,
                        help="The port the OSC server is listening on")
    args = parser.parse_args()

    client = udp_client.SimpleUDPClient(args.ip, args.port)
    odapi = DetectorAPI(relative_path_to_ckpt=args.model,
                        threshold=args.threshold, allowed_movement_per_frame=5, allowed_tracking_loss_frames=10)

    dirname = os.path.dirname(__file__)
    video_path = os.path.join(dirname, args.video)
    capture = cv2.VideoCapture(video_path)
    person_finder = PersonFinder()

    while True:
        r, img = capture.read()
        img = cv2.resize(img, (1280, 720))
        frame_number = int(capture.get(cv2.CAP_PROP_POS_FRAMES))

        (people, scores) = odapi.processFrame(img, frame_number)
        selected_person_name = person_finder.get(people)

        for person in people.values():
            client.send_message("/person/horizontal", person["centroid"][0])
            client.send_message("/person/vertical", person["centroid"][1])

        # FIXME: storing this on the object is bad because it mutates the objects passed by detection
        # FIXME write more functionally... or work out how to pass things around
        for key, person in people.items():
            person["selected"] = key == selected_person_name

        visualise(img, people, scores)
