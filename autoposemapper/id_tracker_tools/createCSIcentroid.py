import glob
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import os


# from autoposemapper.id_tracker_tools import utils


def create_csi_cent(idt_filepath,
                    destination_path=None,
                    auto_source_path=None,
                    abs_tolerance=10,
                    dist_thresh=75,
                    area_thresh=1.1e4,
                    coord_points_path=None,
                    encoder_type='SAE'):
    if auto_source_path is None:
        print('Provide the source path')
        return

    if coord_points_path is None:
        print('provide the name of file to use to look up coordinate information if needed')

    # Find idt files
    idt_file_p = Path(idt_filepath)
    idt_file_s = idt_file_p.stem[:idt_file_p.stem.find('_idtraj')]

    if destination_path is None:
        destination_path = str(idt_file_p.parents[0])
        destination_centroid_file = f'{destination_path}/{idt_file_s}_CSI_centroid.h5'
    else:
        destination_centroid_file = f'{destination_path}/{idt_file_s}_CSI_centroid.h5'

    if os.path.exists(destination_centroid_file):
        return

    # find cnn_auto files
    dfile = glob.glob(f'{auto_source_path}**/*{idt_file_s}_CNN_{encoder_type}.h5', recursive=True)[0]
    dfile_p = Path(dfile)
    dfile_s = dfile_p.stem[:dfile_p.stem.find('_CNN')]

    # check if idt and dlc files are
    if idt_file_s == dfile_s:
        # Load idt files
        print(idt_filepath)
        h5_idt = pd.read_hdf(idt_filepath)
        scorer_idt = h5_idt.columns.get_level_values('scorer').unique().item()
        bodyparts_idt = h5_idt.columns.get_level_values('bodyparts').unique().to_list()
        individuals_idt = h5_idt.columns.get_level_values('individuals').unique().to_list()
        center_idt = h5_idt[scorer_idt]
        center_idt = center_idt.swaplevel(i='individuals', j='bodyparts', axis=1)['Center']

        # Load cnn_auto files
        h5_auto = pd.read_hdf(dfile)
        scorer_auto = h5_auto.columns.get_level_values('scorer').unique().item()
        bodyparts_auto = h5_auto.columns.get_level_values('bodyparts').unique().to_list()
        individuals_auto = h5_auto.columns.get_level_values('individuals').unique().to_list()
        h5_body_auto = h5_auto[scorer_auto][individuals_auto]
        h5_body_auto = h5_body_auto.swaplevel(i='individuals',
                                              j='bodyparts',
                                              axis=1).loc[:, ['betweenEars_midBody', 'midBody_midHip']]
        center_auto = h5_body_auto.betweenEars_midBody.add(h5_body_auto.midBody_midHip).divide(2)

        # Check matching idt and auto individuals
        if np.isnan(center_idt['ind1'].loc[0, 'x']):
            ind_notnull1 = center_idt['ind1'].dropna().index[0]
        else:
            ind_notnull1 = 0

        if np.isnan(center_idt['ind2'].loc[0, 'x']):
            ind_notnull2 = center_idt['ind2'].dropna().index[0]
        else:
            ind_notnull2 = 0

        if np.isclose(center_idt['ind1'].loc[ind_notnull1, 'x'], center_auto['ind1'].loc[ind_notnull1, 'x'], atol=20):
            idt_ind1 = 'ind1'
            auto_ind1 = 'ind1'
            idt_ind2 = 'ind2'
            auto_ind2 = 'ind2'
        elif np.isclose(center_idt['ind2'].loc[ind_notnull2, 'x'], center_auto['ind1'].loc[ind_notnull2, 'x'], atol=20):
            idt_ind1 = 'ind2'
            auto_ind1 = 'ind1'
            idt_ind2 = 'ind1'
            auto_ind2 = 'ind2'
        elif np.isclose(center_idt['ind1'].loc[ind_notnull1, 'x'], center_auto['ind2'].loc[ind_notnull1, 'x'], atol=20):
            idt_ind1 = 'ind1'
            auto_ind1 = 'ind2'
            idt_ind2 = 'ind2'
            auto_ind2 = 'ind1'
        else:
            idt_ind1 = 'ind1'
            auto_ind1 = 'ind1'
            idt_ind2 = 'ind2'
            auto_ind2 = 'ind2'

        #         area_1 = utils.cal_animal_area(h5_auto)
        #         bad_area1 = np.where(area_1.values < area_1.median().item() - area_thresh)[0]
        #         bad_area2 = np.where(area_1.values > area_1.median().item() + area_thresh)[0]
        #         bad_area = np.concatenate((bad_area1, bad_area2))
        #         bad_area = np.unique(bad_area)

        #         bp_angle2 = utils.cal_dist_angle_center(idt_filepath, center_path=coord_points_path,
        #         individual=auto_ind2,
        #                                                scorer='ID_Tracker')
        #         bp_angle2 = bp_angle2['center_angle'].values

        #         coord_points = pd.read_csv(coord_points_path)
        #         coord_points.rename(columns={'Unnamed: 0':'Filename'}, inplace=True)
        #         coord_points.set_index('Filename', inplace=True)

        #         for coord_names in coord_points.index:
        #             if dfile_s in coord_names:
        #                 vname = coord_names
        #                 coord_loc = coord_points.loc[vname,f'Center_x':f'Center_y']
        #                 break

        center_idt1 = center_idt[idt_ind1].values
        center_idt2 = center_idt[idt_ind2].values
        center_auto1 = center_auto[auto_ind1].values
        center_auto2 = center_auto[auto_ind2].values

        animal_auto_idt1 = np.zeros(center_auto1.shape)
        animal_auto_idt2 = np.zeros(center_auto2.shape)

        for i in tqdm(range(center_auto.shape[0])):
            if not np.isnan(center_idt1[i]).any():
                animal_auto_idt1[i] = center_idt1[i]
                animal_auto_idt2[i] = center_idt2[i]
            else:
                animal_auto_idt1[i] = center_auto1[i]
                animal_auto_idt2[i] = center_auto2[i]

        #         animal1 = np.zeros(center_auto1.shape)
        #         animal2 = np.zeros(center_auto2.shape)

        #         animal1[0:3] = animal_auto_idt1[0:3]
        #         animal2[0:3] = animal_auto_idt2[0:3]

        #         for i in tqdm(range(2, animal_auto_idt1.shape[0])):
        #             dist1 = np.diff((animal_auto_idt1[i], animal1[i-1]))
        #             dist2 = np.diff((animal_auto_idt2[i], animal2[i-1]))
        #             dist_previous1 = np.diff((animal1[i-1], animal1[i-2]))
        #             dist_previous2 = np.diff((animal2[i-1], animal2[i-2]))
        #             dist_preprev1 = np.diff((animal1[i-2], animal1[i-3]))
        #             dist_preprev2 = np.diff((animal2[i-2], animal2[i-3]))

        #             pre_diff1 = dist_previous1 - dist_preprev1
        #             pre_diff2 = dist_previous2 - dist_preprev2

        #             cur_diff1 = dist1 - dist_previous1
        #             cur_diff2 = dist2 - dist_previous2

        #             if np.isclose(pre_diff1, cur_diff1, atol=abs_tolerance).any() and
        #             np.isclose(pre_diff2, cur_diff2, atol=abs_tolerance).any():
        #                 animal1[i] = animal_auto_idt1[i]
        #                 animal2[i] = animal_auto_idt2[i]
        #             elif np.isclose(pre_diff1, cur_diff1, atol=abs_tolerance).any():
        #                 animal1[i] = animal_auto_idt1[i]
        #                 animal2[i] = animal_auto_idt2[i]
        #             elif np.isclose(pre_diff2, cur_diff2, atol=abs_tolerance).any():
        #                 animal1[i] = animal_auto_idt1[i]
        #                 animal2[i] = animal_auto_idt2[i]
        #             else:
        #                 animal1[i] = animal_auto_idt2[i]
        #                 animal2[i] = animal_auto_idt1[i]

        # dist1 = utils.cal_df2f(animal_auto_idt1[i], animal1[i-1])
        #             angle = utils.cal_dac(animal1[i-1], coord_loc.values)
        #             if dist1 < dist_thresh:
        #                 animal1[i] = animal_auto_idt1[i]
        #                 animal2[i] = animal_auto_idt2[i]
        #             elif dist2 < dist_thresh:
        #                 animal1[i] = animal_auto_idt1[i]
        #                 animal2[i] = animal_auto_idt2[i]
        #             elif dist_previous1 < dist_thresh:
        #                 animal1[i] = animal_auto_idt1[i]
        #                 animal2[i] = animal_auto_idt2[i]
        #             elif dist_previous2 < dist_thresh:
        #                 animal1[i] = animal_auto_idt1[i]
        #                 animal2[i] = animal_auto_idt2[i]
        #             elif np.isclose(bp_angle2[i], angle, atol=abs_tolerance):
        #                 animal1[i] = animal_auto_idt2[i]
        #                 animal2[i] = animal_auto_idt1[i]
        #             else:
        #                 animal1[i] = animal_auto_idt2[i]
        #                 animal2[i] = animal_auto_idt1[i]

        # animal = np.stack((animal1, animal2),axis=1)
        # animal = animal.reshape(-1, animal.shape[1]*animal.shape[2])

        animal = np.stack((animal_auto_idt1, animal_auto_idt2), axis=1)
        animal = animal.reshape(-1, animal.shape[1] * animal.shape[2])

        col = pd.MultiIndex.from_product([['CSI_Tracker'], individuals_idt, bodyparts_idt, ['x', 'y']],
                                         names=['scorer', 'individuals', 'bodyparts', 'coords'])

        ind_auto_idt = center_auto.index
        animal_df = pd.DataFrame(animal, index=ind_auto_idt, columns=col)

        animal_df.to_hdf(destination_centroid_file, 'vole_d')
