
import argparse
import numpy as np
import tensorflow as tf
import cv2
from detector import DetectorAPI
from pythonosc import osc_message_builder
from pythonosc import udp_client


def visualise(img, people):
    for person in people:
        box = person["image_scaled_box"]
        print(box)
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
        cv2.circle(img, person["image_scaled_centroid"], 10, (0, 255, 0), -1)

    cv2.imshow("preview", img)
    cv2.moveWindow("preview", 0, 0)
    # FIXME this probably shouldnt be here, but capture doesnt work without it
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="../sample_videos/park_day.mp4")
    parser.add_argument(
        "--model", default="./faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb")
    parser.add_argument("--threshold", default=0.7)
    parser.add_argument("--ip", default="192.168.1.108",
                        help="The ip of the OSC server")
    parser.add_argument("--port", type=int, default=8060,
                        help="The port the OSC server is listening on")
    args = parser.parse_args()

    client = udp_client.SimpleUDPClient(args.ip, args.port)
    odapi = DetectorAPI(path_to_ckpt=args.model, threshold=args.threshold)
    capture = cv2.VideoCapture(args.video)

    while True:
        r, img = capture.read()
        img = cv2.resize(img, (1280, 720))

        detections = odapi.processFrame(img)

        for detection in detections:
            client.send_message("/person/horizontal", detection["centroid"][0])
            client.send_message("/person/vertical", detection["centroid"][1])

        cv2.imshow("preview", img)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

        visualise(img, detections)
