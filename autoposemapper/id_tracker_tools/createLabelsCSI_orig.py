import pandas as pd
import glob
import numpy as np
from pathlib import Path
from tqdm import tqdm
from autoposemapper.id_tracker_tools import utils
import os


def create_labels_csi(idt_file, destination_path=None, auto_source_path=None,
                      abs_tolerance=40, tracker='_CSI', encoder_type='SAE'):

    if auto_source_path is None:
        print('Provide the source path to the autoencoder tracked files')
        return

    # Find csi files
    idt_file_p = Path(idt_file)
    idt_file_s = idt_file_p.stem[:idt_file_p.stem.find(tracker)]

    if destination_path is None:
        destination_path = str(idt_file_p.parents[0])
        destination_file = f'{destination_path}/{idt_file_s}_labels.h5'
    else:
        destination_file = f'{destination_path}/{idt_file_s}_labels.h5'

    if os.path.exists(destination_file):
        return

    # find cnn_auto files
    dfile = glob.glob(f'{auto_source_path}**/*{idt_file_s}_CNN_{encoder_type}.h5', recursive=True)[0]
    dfile_p = Path(dfile)
    dfile_s = dfile_p.stem[:dfile_p.stem.find('_CNN')]

    # check if csi and auto files are
    if idt_file_s == dfile_s:
        # Load csi files
        print(idt_file)
        h5_csi = pd.read_hdf(idt_file)
        scorer_csi = h5_csi.columns.get_level_values('scorer').unique().item()
        bodyparts_csi = h5_csi.columns.get_level_values('bodyparts').unique().to_list()
        individuals_csi = h5_csi.columns.get_level_values('individuals').unique().to_list()
        # The centroid for animal 1
        csi_center_0 = h5_csi[scorer_csi][individuals_csi[0]][bodyparts_csi[0]].values

        # The centroid for animal 2
        csi_center_1 = h5_csi[scorer_csi][individuals_csi[1]][bodyparts_csi[0]].values

        # Load auto files
        h5_auto = pd.read_hdf(dfile)
        scorer_auto = h5_auto.columns.get_level_values('scorer').unique().item()
        bodyparts_auto = h5_auto.columns.get_level_values('bodyparts').unique().to_list()
        individuals_auto = h5_auto.columns.get_level_values('individuals').unique().to_list()
        h5_body_auto = h5_auto[scorer_auto][individuals_auto]

        h5_body_0 = h5_auto[scorer_auto][individuals_auto[0]].values
        h5_body_1 = h5_auto[scorer_auto][individuals_auto[1]].values

        center_0 = h5_auto[scorer_auto][individuals_auto[0]].loc[:, ['betweenEars_midBody', 'midBody_midHip']]
        center_0 = center_0.betweenEars_midBody.add(center_0.midBody_midHip).divide(2).values

        center_1 = h5_auto[scorer_auto][individuals_auto[1]].loc[:, ['betweenEars_midBody', 'midBody_midHip']]
        center_1 = center_1.betweenEars_midBody.add(center_1.midBody_midHip).divide(2).values

        labels_0 = np.zeros((center_0.shape[0], 1))
        labels_1 = np.zeros((center_0.shape[0], 1))

        for i in tqdm(range(center_0.shape[0])):
            dist = utils.cal_df2f(csi_center_0[i], center_0[i])
            if dist < abs_tolerance:
                labels_0[i] = 0
                labels_1[i] = 1
            else:
                labels_0[i] = 1
                labels_1[i] = 0

                # Check if any of the body parts are close the ID tracker centroid for vole 1
                for bp in range(len(bodyparts_auto)):
                    bp_value = h5_body_0[i, (bp * 2):(bp * 2) + 2]
                    dist_pb = utils.cal_df2f(csi_center_0[i], center_0[i])
                    if dist_pb < abs_tolerance:
                        labels_0[i] = 0
                        labels_1[i] = 1
                        # check_key = True
                        break

                # Check if any of the body parts are close the ID tracker centroid for vole 2
                # This part takes care of if one centroid tracking is wrong and the other is right
                for bp in range(len(bodyparts_auto)):
                    bp_value = h5_body_1[i, (bp * 2):(bp * 2) + 2]
                    dist_pb = utils.cal_df2f(csi_center_1[i], center_1[i])
                    if dist_pb < abs_tolerance:
                        labels_0[i] = 0
                        labels_1[i] = 1
                        # check_key = True
                        break
                # if not check_key:
                #    labels_0[i] = labels_0[i-1]
                #    labels_1[i] = labels_0[i-1]

        print(destination_file)
        labels_h5 = pd.DataFrame(labels_0, columns=['labels'])
        labels_h5.to_hdf(destination_file, 'vole_d')
