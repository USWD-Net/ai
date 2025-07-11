from glob import glob
import pandas as pd
import os

def get_merged_data(data_path) :
    data_flist = sorted(glob(os.path.join(data_path, "*.csv")))

    df = pd.read_csv(data_flist[0])

    for i in range(1, len(data_flist)) :
        fname = data_flist[i]
        temp = pd.read_csv(fname)
        df = pd.concat([df, temp])
        print('Load & Concat -', fname)

    print('Total Data Length : ',len(df))
    
    return df
