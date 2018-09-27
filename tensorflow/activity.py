from collections import deque
import math
import numpy as np

# frame difference per person (average above threshold = active)
# centroid movement per person (average above threshold = active)
# box change per person

# number of active people - compared to rolling average


# number of people in scene
# frame difference in scene


class Activity():
    def __init__(self):
        average_buffer = 60  # TODO: move this into a config easy for user to set
        self.previous_frames_queue = deque(maxlen=average_buffer)
        self.previous_bounding_boxes = deque(maxlen=average_buffer)

        self.centroids = {}
        self.centroid_movements = {}
        return

    # def store_person_history(self, name, type, data):
    #     if not self.centroids[name]:
    #         setattr(self, type, deque(maxlen=60))
    #     getattr(name, type).append(data)

    # def get_history(self, name, type, distance):
    #     getattr(name, type)[:-distance-1:-1]

    # def calculate_centroid_movement(self, centroid, prev_centroid):
    #     x, y = centroid
    #     prevX, prevY = prev_centroid
    #     movement_since_last_frame = math.hypot(x - prevX, y - prevY)
    #     return movement_since_last_frame

    # def centroid_movement_for_person(self, name, person):
    #     store_person_history(name, 'centroids', person['centroid'])
    #     centroid_movement = self.calculate_centroid_movement(
    #             person["centroid"], get_history(name, 'centroids'))
    #     store_person_history(name, 'centroids', person['centroid'])
    #     store_person_history(name, 'centroid_movements', centroid_movements])
    #     return

    # def frame_difference_for_person(self, img, prev_image, person):
    #     # store frame diff from frame to frame
    #     return

    def box_perimeter(self, box):
        perimeter = (box[2] - box[0]) * 2 + (box[3] - box[1]) * 2
        return perimeter

    def box_area(self, box):
        area = (box[2] - box[0]) * (box[3] - box[1])
        return area

    def box_movement_per_person(self, person):
        # store movement from frame to frame
        return

    def get(self, people):
        return {
            "box_score": self.average_bounding_box(people),
            "people_score": self.average_number_of_people(people)
        }

    def cumulative_movement(self, name, num_frames):
        return sum(get_history(name, num_frames))

    # SCENE AVERAGES
    def most_active_person(self, people):
        sorted([(person, cumulative_movement(person))
                for person in people], key=lambda person: person[1])[0]

    def movement_activity_in_scene(self):
        return sum([cumulative_movement(person, 100) for person in people]) - sum([cumulative_movement(person, 10) for person in people])

    # get the diff between now and rolling average
    def average_number_of_people(self, people):
        current_number_of_people = len(people)
        self.previous_frames_queue.append(current_number_of_people)
        average_number_of_people = np.mean(self.previous_frames_queue)
        return current_number_of_people - average_number_of_people

    # get diff between now and rolling average of box sizes
    def average_bounding_box(self, people):
        total_bounding_box = sum([self.box_area(person["image_scaled_box"])
                                  for k, person in people.items()])
        self.previous_bounding_boxes.append(total_bounding_box)
        average_bounding_box = np.mean(self.previous_bounding_boxes)
        return total_bounding_box - average_bounding_box

    def bounding_box_wave(self, people):
        # get an array of arrays of diff between frame values
        bounding_box_changes = self.get_bounding_box_changes()
        largest_

    def cumulative_centroid_movement(self):
        return sum([sum(person_total) for person_total in self.centroid_movements.values()])
