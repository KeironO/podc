import os
import imageio

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
    