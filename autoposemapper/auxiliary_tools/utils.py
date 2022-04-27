import os
import pandas as pd
import glob


def check_pandas_h5(file_path):
    """
    Check the pandas h5 file
    param file_path: the path to the h5 file(s)
    return: print the head of the h5 file
    """

    if os.path.isfile(file_path):
        h5 = pd.read_hdf(file_path)
        return h5
    elif os.path.isdir(file_path):
        h5files = sorted(glob.glob(f"{file_path}/*.h5"))
        if len(h5files) == 0:
            print('No h5 file found')
        else:
            for file in h5files:
                h5 = pd.read_hdf(file)
                return h5
