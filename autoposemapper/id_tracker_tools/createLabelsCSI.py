import pandas as pd
import glob
import numpy as np
from pathlib import Path
from tqdm import tqdm
from autoposemapper.id_tracker_tools import utils
import os


def create_labels_csi(idt_file, destination_path=None, auto_source_path=None,
                      abs_tolerance=40, tracker='_CSI', encoder_type='SAE',
                      body_part1='betweenEars_midBody', body_part2='midBody_midHip'):

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

        # Load auto files
        print(dfile)
        h5_auto = pd.read_hdf(dfile)
        scorer_auto = h5_auto.columns.get_level_values('scorer').unique().item()
        bodyparts_auto = h5_auto.columns.get_level_values('bodyparts').unique().to_list()
        individuals_auto = h5_auto.columns.get_level_values('individuals').unique().to_list()

        for k, ind in enumerate(individuals_auto):
            csi_center = h5_csi[scorer_csi][ind][bodyparts_csi[0]].values
            h5_body = h5_auto[scorer_auto][ind].values

            auto_center = h5_auto[scorer_auto][ind].loc[:, [body_part1, body_part2]]
            auto_center = auto_center[body_part1].add(auto_center[body_part2]).divide(2).values

            labels = np.zeros((csi_center.shape[0], 1))

            for i in tqdm(range(csi_center.shape[0])):
                dist = utils.cal_df2f(csi_center[i], auto_center[i])
                if dist < abs_tolerance:
                    labels[i] = k
                else:
                    # Check if any of the body parts are close the ID tracker centroid for vole 1
                    for bp in range(len(bodyparts_auto)):
                        bp_value = h5_body[i, (bp * 2):(bp * 2) + 2]
                        dist_pb = utils.cal_df2f(csi_center[i], bp_value)
                        if dist_pb < abs_tolerance:
                            labels[i] = k
                            break

        print(destination_file)
        labels_h5 = pd.DataFrame(labels, columns=['labels'])
        labels_h5.to_hdf(destination_file, 'animal_d')
