import os
from .video import get_max_frames
from .models import VGG19v1

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