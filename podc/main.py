from data import VideoDataGenerator
from deeplearning import Model
import os
import imageio


names = ["positive", "negative"]
videos_dir = "/home/keo7/Videos"

labels_dict = {}

for n in names:
    fn = "videos of %s sliding sign from reproducibility study" % (n)
    files = os.listdir(os.path.join(videos_dir, fn))
    if n == "negative":
        n = 0
    else:
        n = 1
    for i in files:
        labels_dict[i] = n
    

data_dir = "/home/keo7/Data/slidingsign"
filenames = os.listdir(data_dir)

vdg = VideoDataGenerator(data_dir, filenames, labels_dict, 4, height=224, width=224, featurewise_center=True, rotation_range=1, shear_range=5)

for X, y in vdg:
    print (X.shape)
    break

exit(0)


'''

'''

model = Model().model
model.fit_generator(vdg, epochs=1)
