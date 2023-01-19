import os

def retrieve_frames(path_dir):
    frames_list = []
    for filename in os.listdir(path_dir):
        path = os.path.join(path_dir, filename)
        # path = os.path.join(os.getcwd(), path)
        frames_list.append(path)
    return frames_list