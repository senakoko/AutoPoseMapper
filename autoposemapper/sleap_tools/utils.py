import glob

import numpy as np
import pandas as pd
import h5py
import os
from tqdm import tqdm
from pathlib import Path
from autoposemapper.setRunParameters import set_run_parameter


def convert_sh5_to_ph5(file, destination_path=None, scorer='SLEAP'):
    """
    converts the h5 file generated by sleap to pandas table h5 file format.

    parameters
    ----------
    file: the file to be converted
    destination_path: the path to save the h5 files
    scorer: name of the scorer: default to 'SLEAP'

    """
    parameters = set_run_parameter()
    # Destination filename
    if destination_path is None:
        destination_name = Path(file).stem
        destination_path = str(Path(file).parents[0])
        destination_file = f"{destination_path}/{destination_name}_{parameters.conv_tracker_name}.h5"
    else:
        destination_name = Path(file).stem
        destination_file = f"{destination_path}/{destination_name}_{parameters.conv_tracker_name}.h5"

    if os.path.exists(destination_file):
        return

    with h5py.File(file, "r") as f:
        locations = f["tracks"][:].T
        node_names = [n.decode() for n in f["node_names"][:]]

    body_parts = node_names  # Body parts that were tracked
    animal_number = locations.shape[-1]  # Get the number of animals tracked
    individuals = []  # list for individuals
    coord = 2  # x and y coordinates

    cnn_data = np.zeros((locations.shape[0], locations.shape[1] * coord * animal_number))  # dataframe to store

    for j in tqdm(range(animal_number)):
        individual = f'ind{j + 1}'  # Individual tracked
        anim = locations[:, :, :, j]  # get the location data
        for i in range(anim.shape[1]):
            cnn_data[:, coord * i + (j * len(body_parts) * 2)] = anim[:, i, 0]  # Represent x coordinate
            cnn_data[:, coord * i + 1 + (j * len(body_parts) * 2)] = anim[:, i, 1]  # Represent y coordinate
        individuals.append(individual)

    col = pd.MultiIndex.from_product([[scorer], individuals, body_parts, ['x', 'y']],
                                     names=['scorer', 'individuals', 'bodyparts', 'coords'])
    print(destination_file)

    data_df = pd.DataFrame(cnn_data, columns=col)
    data_df.to_hdf(destination_file, parameters.animal_key)


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
