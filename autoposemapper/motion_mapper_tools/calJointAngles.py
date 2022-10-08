import pandas as pd
import numpy as np
from pathlib import Path
import os
from tqdm import tqdm


def cal_joint_angles(file=None, destination_path=None, encoder_type='CNN'):
    """
    Calculate the joint angles between body parts and returns a dataframe of distances
    for only one animal

    Expects to read h5 files. Doesn't work for csv files

    Parameters
    ----------
    file: the path to file with the tracked points.
    destination_path: the path to save tracked points.
    encoder_type: either SAE, VAE or ANT
    """

    destination_name = Path(file).stem
    destination_name = destination_name[:destination_name.find(encoder_type)]

    name = 'animal_1'
    if destination_path is None:
        destination_path = str(Path(file).parent)
        destination_file = f"{destination_path}/{destination_name}{name}_joint_angles.h5"
    else:
        destination_file = f"{destination_path}/{destination_name}{name}_joint_angles.h5"

    if os.path.exists(destination_file):
        return

    print(file)
    # Read file
    with pd.HDFStore(file) as df:
        animal_key = df.keys()[0][1:]
        h5 = df[animal_key]

    # Get scorer
    scorer = h5.columns.get_level_values('scorer').unique().item()

    # Get list of individuals
    individuals = h5.columns.get_level_values('individuals').unique()

    center = 'midBody'

    vector1_bodyparts = ['Nose', 'midHip']

    vector2_bodyparts = ['leftEar', 'betweenEars', 'rightEar', 'rightMidWaist',
                         'leftMidWaist', 'leftHip', 'rightHip', 'tailStart']

    animals_angles = pd.DataFrame()
    for it, ind in enumerate(individuals):

        ind_h5 = h5[scorer][ind]

        for vec1 in vector1_bodyparts:
            vector1 = ind_h5[center].sub(ind_h5[vec1])
            vector1 = vector1 / np.linalg.norm(vector1, axis=1)[:, np.newaxis]
            for vec2 in vector2_bodyparts:
                vector2 = ind_h5[center].sub(ind_h5[vec2])
                vector2 = vector2 / np.linalg.norm(vector2, axis=1)[:, np.newaxis]
                angle_data = np.zeros((vector2.shape[0], 1))
                for t in tqdm(range(vector2.shape[0])):
                    v1 = vector1.iloc[t]
                    v2 = vector2.iloc[t]
                    angle = np.arccos(np.dot(v1, v2)) * 180 / np.pi
                    angle_data[t] = angle
                animals_angles[f'{vec1}_{vec2}'] = angle_data.flatten()

        name = f'animal_{it + 1}'
        if destination_path is None:
            destination_path = file.rsplit('/', 1)[0]
            destination_file = f"{destination_path}/{destination_name}{name}_joint_angles.h5"
        else:
            destination_file = f"{destination_path}/{destination_name}{name}_joint_angles.h5"

        print(destination_file)
        animals_angles.fillna(method='bfill', inplace=True)
        animals_angles.to_hdf(destination_file, animal_key)

    return animals_angles
