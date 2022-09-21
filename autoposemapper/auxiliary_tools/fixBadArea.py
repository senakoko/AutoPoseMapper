import os
import pandas as pd
import numpy as np
from pathlib import Path
from autoposemapper.auxiliary_tools import utils
from tqdm import tqdm
from autoposemapper.setRunParameters import set_run_parameter


def fix_bad_area(file=None, destination_path=None, tracker='filtered'):
    """
    Fixes the bad tracking where the area of the vole is abnormal

    Expects to read h5 files. Doesn't work for csv files

    Parameters
    ----------
    file: the file with the original tracked points
    destination_path: the path to save tracked points.
    tracker: the name to append to the file
    """

    # Reads the path to the h5 files
    parameters = set_run_parameter()
    file_p = Path(file)
    parent = file_p.parents[0]
    new_name = file_p.stem + f'_{tracker}.h5'

    if destination_path is None:
        destination_file = f"{parent}/{new_name}"
    else:
        destination_file = f"{str(destination_path)}/{new_name}"

    if os.path.exists(destination_file):
        return

    print(destination_file)

    h5 = pd.read_hdf(file)
    scorer = h5.columns.get_level_values('scorer').unique().item()
    individuals = h5.columns.get_level_values('individuals').unique().to_list()
    bodyparts = h5.columns.get_level_values('bodyparts').unique().to_list()
    animal_main_df = pd.DataFrame()

    for ind in individuals:
        area = utils.cal_animal_area(h5, scorer=scorer, individual=ind)
        area_thresh = (area.median() * 0.5).item()
        bad_area1 = np.where(area.values < area.median().item() - area_thresh)[0]
        bad_area2 = np.where(area.values > area.median().item() + area_thresh)[0]
        bad_area = np.concatenate((bad_area1, bad_area2))
        bad_area = np.unique(bad_area)
        h5_body = h5[scorer][ind].values
        animal = np.zeros(h5_body.shape)
        for i in tqdm(range(h5_body.shape[0])):
            if i == 0:
                animal[i] = h5_body[i]
            elif i in bad_area:
                # print(i)
                animal[i] = animal[i - 1]
            else:
                animal[i] = h5_body[i]
        animal_df = pd.DataFrame(animal)
        animal_main_df = pd.concat([animal_main_df, animal_df], ignore_index=True, axis=1)

    col = pd.MultiIndex.from_product([[scorer], individuals, bodyparts, ['x', 'y']],
                                     names=['scorer', 'individuals', 'bodyparts', 'coords'])

    ind_values = h5.index
    dataframe = pd.DataFrame(animal_main_df.values, index=ind_values, columns=col)
    dataframe.to_hdf(destination_file, parameters.animal_key)
