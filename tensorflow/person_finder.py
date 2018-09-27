import random
from activity import Activity

# FIXME uses the name to identify people which is eventually recycled - use a unique ID?


class PersonFinder:
    def __init__(self):
        self.track_for_frames = 50  # FIXME: move to config
        self.selected_person = None
        self.ttl = self.track_for_frames
        self.activity = Activity()

    def select_random(self, people):
        return random.choice(list(people.keys()))

    def select_active(self, people):
        return self.activity.most_active_person(people)

    def get(self, people):
        if not self.selected_person or self.selected_person not in people or self.ttl <= 0:
            # FIXME should be 0.5 make in config
            self.selected_person = self.select_random(
                people) if random.random() > 0.5 else self.select_active(people)
            self.ttl = self.track_for_frames
            return self.selected_person
        else:
            self.ttl -= 1
            return self.selected_person
