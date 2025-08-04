def predicate_set():
    mental_predicate_set = [1, 2, 7, 8, 9, 10, 11]
    action_predicate_set = [3, 4, 5, 6]
    head_predicate_set = [1, 2, 3, 4]
    total_predicate_set = [1, 2, 3, 4]
    return mental_predicate_set, action_predicate_set, head_predicate_set, total_predicate_set

mental_predicate_set, action_predicate_set, head_predicate_set, total_predicate_set = predicate_set()
PAD = 0

time_horizon = 15
grid_length = 0.50
