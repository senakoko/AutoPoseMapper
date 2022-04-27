import numpy as np
import pandas as pd
from pathlib import Path


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
    bodypart = h5[scorer][individual].loc[:, ['betweenEars_midBody', 'midBody_midHip']]
    center = bodypart.betweenEars_midBody.add(bodypart.midBody_midHip).divide(2)
    a_len = nose.sub(center)
    b_len = center.sub(left_mid)
    a_dist = np.linalg.norm(a_len, axis=1)
    b_dist = np.linalg.norm(b_len, axis=1)
    area = np.pi * a_dist * b_dist
    area_df = pd.DataFrame({'area': area})

    return area_df


def cal_dist_f2f(h5_data=None, scorer="Stacked_Autoencoder", individual='ind1', ):
    """
    Calculate the euclidean distances between frames

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

    if scorer == 'Stacked_Autoencoder':
        bodypart = h5[scorer][individual].loc[:, ['betweenEars_midBody', 'midBody_midHip']]
        center = bodypart.betweenEars_midBody.add(bodypart.midBody_midHip).divide(2)
    else:
        center = h5[scorer][individual]['Center']

    diff = center.diff(axis=0)
    dist = np.linalg.norm(diff, axis=1)
    dist_df = pd.DataFrame({'dist': dist})
    dist_df = dist_df.fillna(method='bfill')

    return dist_df


def cal_df2f(value1, value2):
    diff = value1 - value2
    dist = np.linalg.norm(diff)
    return dist


def cal_dac(value1, value2):
    diff = value1 - value2
    angle = np.arctan2(diff[1], diff[0]) * 180 / np.pi
    return angle


def cal_dist_angle_center(h5_data=None, scorer="Stacked_Autoencoder", individual='ind1', center_path=None, file_p=None):
    """
    Calculate the euclidean distances and angle to the center of the cage

    Parameters
    ----------
    h5_data: h5 data or path to h5 data
    scorer: the annotator/scorer of h5 file
    individual: which individual to calculate its area
    center_path: path to the file with coordinates about the center of the cage
    file_p: Name of file to use to look up coordinate information
    """

    if not isinstance(h5_data, pd.DataFrame):
        h5 = pd.read_hdf(h5_data)
        file = Path(h5_data)
        if scorer == 'Stacked_Autoencoder':
            file_p = file.stem[:file.stem.find('_CNN')]
        else:
            file_p = file.stem[:file.stem.find('_idtr')]
    else:
        h5 = h5_data

    if center_path is None:
        print('provide path to the file with coordinates about the center of the cage')
        return

    if file_p is None:
        print('provide the name of file to use to look up coordinate information')
        return

    coord_points = pd.read_csv(center_path)
    coord_points.rename(columns={'Unnamed: 0': 'Filename'}, inplace=True)
    coord_points.set_index('Filename', inplace=True)

    for coord_names in coord_points.index:
        if file_p in coord_names:
            vid_name = coord_names
            coord_loc = coord_points.loc[vid_name, f'Center_x':f'Center_y']
            break

    if scorer == 'Stacked_Autoencoder':
        bodypart = h5[scorer][individual].loc[:, ['betweenEars_midBody', 'midBody_midHip']]
        center = bodypart.betweenEars_midBody.add(bodypart.midBody_midHip).divide(2)
    else:
        center = h5[scorer][individual]['Center']

    bp_dist = center.sub(coord_loc.values)
    bp_dist_angle = pd.DataFrame()
    bp_dist_angle[f'center_dist'] = np.linalg.norm(bp_dist, axis=1)
    bp_dist_angle[f'center_angle'] = np.arctan2(bp_dist['y'], bp_dist['x']) * 180 / np.pi

    return bp_dist_angle


