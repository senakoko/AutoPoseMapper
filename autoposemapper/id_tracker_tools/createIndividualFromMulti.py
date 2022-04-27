import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def create_individuals_4_multi(file=None, destination_path=None, tracker='CSI'):
    """
    Creates individual from multi-animal tracking
    Expects to read h5 files. Doesn't work for csv files

    Parameters
    ----------
    file: the file with the original tracked points.
    destination_path: the path to save tracked points.
    tracker: the autoencoder file
    """

    # Reads the path to the h5 files
    file_p = Path(file)
    parent = file_p.parents[0]
    new_name = file_p.stem[:file_p.stem.find(f'_{tracker}')] + f'_{tracker}_animal_1_data.h5'

    if destination_path is None:
        destination_file = f"{parent}/{new_name}"
    else:
        destination_file = f"{str(destination_path)}/{new_name}"

    if os.path.exists(destination_file):
        return

    print(file)

    h5 = pd.read_hdf(file)
    scorer = h5.columns.get_level_values('scorer').unique().item()
    individuals = h5.columns.get_level_values('individuals').unique().to_list()
    bodyparts = h5.columns.get_level_values('bodyparts').unique().to_list()

    for i, ind in tqdm(enumerate(individuals)):
        file_p = file[:file.find(tracker)]
        destination_file = file_p + f'{tracker}_animal_{i + 1}_data.h5'
        if not os.path.exists(destination_file):
            animal = h5[scorer][ind]
            col = pd.MultiIndex.from_product([[scorer], bodyparts, ['x', 'y']],
                                             names=['scorer', 'bodyparts', 'coords'])
            dataframe = pd.DataFrame(animal.values, index=animal.index, columns=col)
            dataframe.to_hdf(destination_file, 'animal_d')
