import os
import pandas as pd
import glob
import numpy as np


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


def cal_animal_area(h5_data=None, scorer="Stacked_Autoencoder", individual='ind1'):
    """
    Calculate the area of the animal assuming it has an ellipsoid shape

    Parameters
    ----------
    h5_data: h5 data or path to h5 data
    scorer: the annotator/scorer of h5 file
    individual: which individual to calculate its area
    """
    if not isinstance(h5_data, pd.DataFrame):
        h5 = pd.read_hdf(h5_data)
    else:
        h5 = h5_data

    nose = h5[scorer][individual].loc[:, 'Nose']
    left_mid = h5[scorer][individual].loc[:, 'leftMidWaist']
    bodypart = h5[scorer][individual].loc[:, ['leftMidWaist', 'rightMidWaist']]
    center = bodypart.leftMidWaist.add(bodypart.rightMidWaist).divide(2)
    a_len = nose.sub(center)
    b_len = center.sub(left_mid)
    a_dist = np.linalg.norm(a_len, axis=1)
    b_dist = np.linalg.norm(b_len, axis=1)
    area = np.pi * a_dist * b_dist
    area_df = pd.DataFrame({'area': area})

    return area_df
