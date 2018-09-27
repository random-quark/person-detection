import time
import argparse
import os
import cv2
from pythonosc import udp_client
from detector import DetectorAPI
from activity import Activity

from config import config

from person_finder import PersonFinder


def visualise(img, people, scores, selected_person_name):
    im_height, _, _ = img.shape
    for name, person in people.items():
        box = person["image_scaled_box"]
        color = (0, 0, 255) if name == selected_person_name else (0, 255, 0)
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
        cv2.circle(img, person["image_scaled_centroid"], 10, color, -1)
        textCoords = tuple(
            [pos - 10 for pos in person["image_scaled_centroid"]])
        cv2.putText(img, person["name"], (textCoords),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color)

    cv2.putText(img, "Total people: {}".format(
        scores["total_people"]), (10, im_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
    cv2.putText(img, "Avg num people: {}".format(
        scores["average_number_people"]), (10, im_height - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
    cv2.putText(img, "Deviation from avg: {}".format(
        scores["activity_score"]), (10, im_height - 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))

    cv2.imshow("preview", img)
    cv2.moveWindow("preview", 0, 0)

    # FIXME this probably shouldnt be here, but capture doesnt work without it
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        return


if __name__ == "__main__":
    client = udp_client.SimpleUDPClient(
        config["osc_server_ip"], config["osc_server_port"])
    odapi = DetectorAPI(relative_path_to_ckpt=config["model_path"])

    # dirname = os.path.dirname(__file__)
    # video_path = os.path.join(dirname, args.video)

    capture = cv2.VideoCapture(config["video_source"])
    person_finder = PersonFinder()
    activity = Activity()

    last_activity_score_send_time = time.time()

    while True:
        if not capture.isOpened():
            print("ERROR: camera or video selected not available")
            continue

        r, img = capture.read()
        img = cv2.resize(img, (1280, 720))
        frame_number = int(capture.get(cv2.CAP_PROP_POS_FRAMES))

        people = odapi.processFrame(img, frame_number)
        scores = activity.update_and_get(people.copy())
        selected_person_name = person_finder.get(people)

        if selected_person_name:
            selected_person = people[selected_person_name]
            client.send_message("/person/horizontal",
                                selected_person["centroid"][0])
            client.send_message("/person/vertical",
                                selected_person["centroid"][1])
        visualise(img, people, scores, selected_person_name)

        client.send_message('/people/total', scores["total_people"])
        client.send_message('/average_number_people',
                            scores["average_number_people"])

        if time.time() - last_activity_score_send_time > config["activity_score_send_interval"]:
            client.send_message('/activity_score', scores["activity_score"])
            last_activity_score_send_time = time.time()
