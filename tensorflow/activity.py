from collections import deque
import math
import numpy as np
import itertools

from config import config


class Activity():
    instance = None

    def __init__(self):
        if not Activity.instance:
            Activity.instance = Activity.__Activity()

    def __getattr__(self, name):
        return getattr(self.instance, name)

    class __Activity():
        def __init__(self):
            self.previous_frames_queue = deque(
                maxlen=config["people_lookback_frames"])

        # DATA
        def store_person_history(self, name, type, data):
            # sets self[name][type].append(data)
            if not getattr(self, name, None):
                setattr(self, name, {})
            if not getattr(self, name).get(type, None):
                getattr(self, name)[type] = []
            getattr(self, name)[type].append(data)

        def get_history(self, name, type, distance):
            history = getattr(self, name, {}).get(type, [])
            if history:
                return history[: - distance - 1:-1]
            return history

        # CACLULATIONS
        def calculate_centroid_movement(self, centroid, prev_centroid):
            x, y = centroid
            prevX, prevY = prev_centroid
            movement_since_last_frame = math.hypot(x - prevX, y - prevY)
            return movement_since_last_frame

        def centroid_movement_for_person(self, name, person):
            centroids = self.get_history(name, 'centroids', 1)
            if len(centroids) < 1:
                return
            prev_centroid = centroids[0]
            self.store_person_history(name, 'centroids', person['centroid'])
            centroid_movement = self.calculate_centroid_movement(
                person['centroid'], prev_centroid)
            self.store_person_history(name, 'centroids', person['centroid'])
            self.store_person_history(
                name, 'centroid_movements', centroid_movement)

        def cumulative_movement(self, name):
            return sum(self.get_history(name, 'centroid_movements', config["movement_lookback_frames"]))

        # SCENE TOTALS
        def most_active_person(self, people):
            person_by_movement = [(person, self.cumulative_movement(person))
                                  for person in people]
            return sorted(person_by_movement)[0][0]

        # get the diff between now and rolling average
        def activity_score(self, people):
            current_number_of_people = len(people)
            self.previous_frames_queue.append(current_number_of_people)
            average_number_of_people = np.mean(self.previous_frames_queue)
            return current_number_of_people - average_number_of_people

        # get the average number of people over last n frames
        def average_number_of_people(self, people):
            current_number_of_people = len(people)
            self.previous_frames_queue.append(current_number_of_people)
            average_number_of_people = np.mean(self.previous_frames_queue)
            return average_number_of_people

        # EXTERNAL FUNCTIONS
        def update_and_get(self, people):
            for name, person in people.items():
                self.centroid_movement_for_person(name, person)

            return {
                "total_people": len(people),
                "average_number_people": self.average_number_of_people(people),
                "activity_score": self.activity_score(people)
            }
