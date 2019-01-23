import pandas as pd

class Mias():
    pass

def get_metadata(fp):
    df = pd.read_csv(fp, index_col=0)
    return df

get_metadata("/home/keo7/Data/MIAS/Info.txt")