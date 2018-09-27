config = {
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
    "selection_type_ratio": 0.5
}
