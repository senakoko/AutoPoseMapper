import numpy as np
import pandas as pd
import re
from pathlib import Path
import os
import glob


def combine_h5_files(file, destination_path=None, encoder_type='SAE'):
    """
    Keep specific bodyparts from deeplabcut

    Expects to read h5 files. Doesn't work for csv files

    Parameters
    ----------
    file: the file with the tracked points.
    destination_path: the path to save kept body part points
    encoder_type: the autoencoder type used to filter the data either SAE or VAE
    """
    # read h5 file
    #     print(file, end='\n'*2)

    # Destination filename
    destination_name = Path(file).stem
    if re.search(f'_{encoder_type}_', destination_name):
        destination_name = destination_name[:destination_name.find(f'_{encoder_type}_')]
    if destination_path is None:
        destination_path = file.rsplit('/', 1)[0]
        destination_file = f"{destination_path}/{destination_name}_CNN_{encoder_type}.h5"
    else:
        destination_file = f"{destination_path}{destination_name}_CNN_{encoder_type}.h5"

    if os.path.exists(destination_file):
        return

    part_file = file[:file.find('_animal_')]
    auto_files = sorted(glob.glob(f'{part_file}*.h5', recursive=True))
    num_ind = len(auto_files)  # the number of individuals
    individuals = []

    for ind_num in range(num_ind):
        individual = f'ind{ind_num+1}'
        individuals.append(individual)

    coord = 2  # x and y coordinates

    h5 = pd.read_hdf(file)

    # Get scorer
    scorer = h5.columns.get_level_values('scorer').unique().item()

    body_parts = h5.columns.get_level_values('bodyparts').unique().to_list()

    # Initialize combine data points
    combine_data = np.zeros((h5.shape[0], len(body_parts) * coord * num_ind))

    for j, file in enumerate(auto_files):

        print(file, end='\n' * 2)

        h5 = pd.read_hdf(file)

        # Loop to filter the data
        for i, bp in enumerate(body_parts):
            body_part = h5[scorer][bp]

            # combine body parts
            combine_data[:, coord * i + (j * len(body_parts) * coord)] = body_part.values[:, 0]
            combine_data[:, coord * i + 1 + (j * len(body_parts) * coord)] = body_part.values[:, 1]

    col = pd.MultiIndex.from_product([[scorer], individuals, body_parts, ['x', 'y']],
                                     names=['scorer', 'individuals', 'bodyparts', 'coords'])

    combine_data_df = pd.DataFrame(combine_data, index=h5.index, columns=col)
    combine_data_df.where(combine_data_df > 0, inplace=True)
    combine_data_df.interpolate(inplace=True)

    combine_data_df.to_hdf(destination_file, 'animal_d')
