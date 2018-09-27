import random
from activity import Activity
from config import config

# FIXME uses the name to identify people which is eventually recycled - use a unique ID?


class PersonFinder:
    def __init__(self):
        self.track_for_frames = config["person_tracking_period"]
        self.selected_person = None
        self.ttl = self.track_for_frames
        self.activity = Activity()

    def select_random(self, people):
        return random.choice(list(people.keys()))

    def select_active(self, people):
        return self.activity.most_active_person(people)

    def get(self, people):
        if not people:
            return None
        if not self.selected_person or self.selected_person not in people or self.ttl <= 0:
            self.selected_person = self.select_random(
                people) if random.random() > config["selection_type_ratio"] else self.select_active(people)
            self.ttl = self.track_for_frames
            return self.selected_person
        self.ttl -= 1
        return self.selected_person
