import os
import imageio
import matplotlib.pyplot as plt

def get_max_frames(d):
    info = []
    index = 0
    for fn in os.listdir(d):
        filepath = os.path.join(d, fn)
        video = imageio.get_reader(filepath, "ffmpeg")
        meta = video.get_meta_data()
        nframes = meta["nframes"]
        duration = meta["duration"]

        info.append([fn, duration, nframes])
    

def get_labels(d):
    names = ["positive", "negative"]
    labels_dict = {}
    for n in names:
        fn = "videos of %s sliding sign from reproducibility study" % (n)
        files = os.listdir(os.path.join(d, fn))
        if n == "negative":
            n = 0
        else:
            n = 1
        for i in files:
            labels_dict[i] = n

    return labels_dict