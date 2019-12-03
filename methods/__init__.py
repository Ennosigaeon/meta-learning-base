import os

ALGORITHMS = []

if len(ALGORITHMS) == 0:
    directory = os.path.join(os.path.dirname(__file__))
    for file_name in os.listdir(directory):
        if file_name.endswith('.json'):
            ALGORITHMS.append(file_name)
