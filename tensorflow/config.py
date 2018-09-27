config = {
    # range: 0-1. The confidence of the detection algorithm that the detection is a person
    "threshold": 0.7,

    # the percerntage of vertical or horizontal screenspace a person's centroid can move in 1 frame before they become a new person
    "allowed_movement_per_frame": 5,

    # the number of frames that a person can not be detected for before they are deleted from the list
    "allowed_tracking_loss_frames": 10,

    # the number of frames that a person will be tracked before selecting a new person
    "person_tracking_period": 50,

    # seconds, time between sending an OSC message with current level of activity to graphics app
    "activity_score_send_interval": 5,

    # number of frames used to calculate the total movement of a persons centroid in given period to find most active person
    "movement_lookback_frames": 60,

    # number of frames used to calculate the the average number of people in a given period
    "people_lookback_frames": 60,

    # range: 0-1 - the probability of selecting by the most active or selecting by chance
    # 0 will always select a random person
    # 1 will always select the most active person
    # 0.5 means a 50% chance of random selection and 50% chance of selecting by activity
    "selection_type_ratio": 0.5,

    "names_source": ["Tom", "Simon", "Leslie", "John", "Peter", "Ruby", "Sioban", "Ella", "Jane"]
}
