import glob
import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def swap_classified_tracks(file=None, auto_source_path=None,
                           destination_path=None,
                           tracker='CSI_filtered',
                           encoder_type='SAE'):
    """
    Swaps the tracked points based on the classified identities of the animal

    Expects to read h5 files. Doesn't work for csv files

    Parameters
    ----------
    file: the file with the original tracked points.
    auto_source_path: the source to autoencoder tracked files.
    destination_path: the path to save tracked points.
    tracker: the name to append to the file
    encoder_type:  the autoencoder used to filter the tracked points.
    """

    # Reads the path to the h5 files
    if auto_source_path is None:
        print('Provide the source path to the autoencoder tracked files')
        return
    file_p = Path(file)
    file_s = file_p.stem[:file_p.stem.find('_labels')]

    # find cnn_auto files
    auto_file = glob.glob(f'{auto_source_path}**/*{file_s}_CNN_{encoder_type}.h5', recursive=True)[0]

    new_name = Path(auto_file).stem
    new_name = new_name[:new_name.find('_CNN')] + f'_{tracker}.h5'
    if destination_path is None:
        destination_file = f"{str(Path(auto_file).parents[0].resolve())}/{new_name}"
    else:
        destination_file = f"{str(destination_path)}/{new_name}"

    if os.path.exists(destination_file):
        return

    print(file)

    # Loading datasets
    orig_h5 = pd.read_hdf(auto_file)
    labels_h5 = pd.read_hdf(file)

    # Get bodyparts
    bodyparts = orig_h5.columns.get_level_values('bodyparts').unique().to_list()

    # Get scorer 
    scorer = orig_h5.columns.get_level_values('scorer').unique().item()

    individuals = orig_h5.columns.get_level_values('individuals').unique().to_list()

    # Initialize smooth data points
    # coord = 2  # x and y coordinates
    # num_ind = len(individuals)  # the number of individuals

    data_df = pd.DataFrame()
    for i, ind in tqdm(enumerate(individuals)):
        # Get the original data
        anim = orig_h5[scorer][ind]

        data = anim.values

        # Swap the tracked points based on classified identities
        for lv, val in enumerate(labels_h5.values):
            if val == i:
                data[lv] = anim.iloc[lv, :]

        data = pd.DataFrame(data)
        data_df = pd.concat((data_df, data), axis=1, ignore_index=True)

    col = pd.MultiIndex.from_product([[tracker], individuals, bodyparts, ['x', 'y']],
                                     names=['scorer', 'individuals', 'bodyparts', 'coords'])
    data_ind = orig_h5.index
    dataframe = pd.DataFrame(data_df.values, index=data_ind, columns=col)
    print(destination_file)
    dataframe.to_hdf(destination_file, 'animal_d')
    # return dataframe
