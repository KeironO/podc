import pandas as pd
import imageio

class MIAS():
    def __init__(self, image_dir, labels_fp):
        self.image_dir = image_dir
        self.labels_fp = labels_fp

    def load_data(self, c):
        labels = self._get_labels(self.labels_fp)
        labels = labels[c]
        self._get(labels)

    def _get(self, labels):
        for filename, values in labels.iteritems():
            print(values)

    def _get_labels(self, fp):
        df = pd.read_csv(fp, index_col=0)
        return df


if __name__ == "__main__":
    mias = MIAS("/home/kto/Data/MIAS/images", "/home/kto/Data/MIAS/Info.txt")
    print(mias.load_data(["class"]))

#get_metadata("/home/kto/Data/MIAS/Info.txt")