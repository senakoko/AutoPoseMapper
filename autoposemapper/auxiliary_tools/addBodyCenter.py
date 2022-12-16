import pandas as pd
from pathlib import Path
import re


def add_body_center(file, tracker='ANT_filtered'):
    file_p = Path(file)
    parent = file_p.parents[0]
    stem = file_p.stem
    if re.search('_filtered', stem):
        stem = stem[:stem.find('_filtered')]

    new_name = stem + f'_{tracker}.h5'

    destination_file = f"{parent}/{new_name}"

    # Including the 'midBody' in the old list of bodyparts
    new_bodyparts = ['Nose', 'leftEar', 'betweenEars', 'rightEar', 'rightMidWaist', 'midBody',
                     'leftMidWaist', 'leftHip', 'midHip', 'rightHip', 'tailStart']

    with pd.HDFStore(file) as df:
        animal_key = df.keys()[0][1:]
        h5 = df[animal_key]

    scorer = h5.columns.get_level_values('scorer').unique().item()
    individuals = h5.columns.get_level_values('individuals').unique()
    bodyparts = h5.columns.get_level_values('bodyparts').unique()

    if 'midBody' not in bodyparts:

        # Create new h5 file with the body parts to be kept
        main_data = pd.DataFrame()
        for ind in individuals:
            data_ind = h5[scorer][ind]
            bodypart = h5[scorer][ind].loc[:, ['rightMidWaist', 'leftMidWaist']]
            center = bodypart.rightMidWaist.add(bodypart.leftMidWaist).divide(2)
            col_center = pd.MultiIndex.from_product([['midBody'], ['x', 'y']], names=['bodyparts', 'coords'])
            center_df = pd.DataFrame(center.values, index=center.index, columns=col_center)
            new_data = pd.concat([data_ind, center_df], axis=1)

            # Rearrange the column names in the new h5 file
            rearranged_data = pd.DataFrame()
            for bpts in new_bodyparts:
                rearranged_data = pd.concat([rearranged_data, new_data[bpts]], axis=1)
            main_data = pd.concat([main_data, rearranged_data], axis=1)

        # Save processed h5 file
        col = pd.MultiIndex.from_product([[scorer], individuals, new_bodyparts, ['x', 'y']],
                                         names=['scorer', 'individuals', 'bodyparts', 'coords'])
        main_df = pd.DataFrame(main_data.values, index=main_data.index, columns=col)
        main_df.to_hdf(destination_file, animal_key)
        print('Added midBody')

        return main_df

    else:
        print('midBody already added to the file')
