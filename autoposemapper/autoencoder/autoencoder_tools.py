import shutil
import warnings
import glob
import os
from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd

from autoposemapper.autoencoder.egocenter_h5 import egocenter_h5
from autoposemapper.autoencoder.reorient_mat import reorient
from autoposemapper.autoencoder.combineh5files import combine_h5_files
from autoposemapper.setRunParameters import set_run_parameter

import yaml
from scipy.io import savemat, loadmat

warnings.filterwarnings('ignore')


class AutoEncoderHelper:
    def __init__(self, project_path, parameters=None):
        self.project_path = project_path
        self.parameters = parameters

        if self.parameters is None:
            self.parameters = set_run_parameter()

    def egocenter_files(self, bind_center='midBody', b1='Nose',
                        b2='tailStart', drop_point=True, which_points=['tailEnd']):

        h5_path = Path(self.project_path) / self.parameters.autoencoder_data_name
        h5_files = sorted(glob.glob(f'{str(h5_path)}/*.h5'))

        config_path = Path(self.project_path) / self.parameters.config_name
        config_path = str(config_path.resolve())

        for file in h5_files:
            file_p = Path(file).resolve()
            file_s = file_p.stem
            animal_1 = file_p.parents[0] / f'{file_p.stem}_ego_animal_1_data.mat'
            if animal_1.exists():
                print('already processed ', file_s)
                continue
            sub_file_folder = file_p.parents[0] / f'{file_s[:file_s.find(f"_{self.parameters.conv_tracker_name}")]}'
            if not sub_file_folder.exists():
                sub_file_folder.mkdir()
            else:
                print(f'{file_p.stem} already exist')
            destination_file = sub_file_folder / f'{file_p.name}'
            shutil.move(str(file_p), str(destination_file))

            destination_file = str(destination_file)
            animal_1 = destination_file[:destination_file.find(f'_{self.parameters.conv_tracker_name}')]
            file_name = f"{animal_1}_{self.parameters.conv_tracker_name}_ego_animal_1_data.mat"
            if not os.path.exists(file_name):
                print(file)

                bc_value, b1_v, b2_v, body_parts, bpts_val = egocenter_h5(destination_file,
                                                                          bind_center=bind_center, b1=b1,
                                                                          b2=b2, drop_point=drop_point,
                                                                          which_points=which_points)

                cfg_file = {"bind_center_value": bc_value, "b1_value": b1_v, "b2_value": b2_v,
                            "body_parts": body_parts, "body_part_values": bpts_val}
                with open(config_path, 'r') as fr:
                    data = yaml.load(fr, Loader=yaml.FullLoader)
                    if 'bind_center_value' in data.keys():
                        print('bind_center_value already added to config file')
                    else:
                        with open(config_path, 'a') as fw:
                            yaml.dump(cfg_file, fw, default_flow_style=False, sort_keys=False)

    def reorient_files(self, encoder_type='SAE'):

        mat_path = Path(self.project_path) / self.parameters.autoencoder_data_name
        mat_files = sorted(glob.glob(f'{str(mat_path)}/**/*{encoder_type}_ego*.mat', recursive=True))

        config_path = Path(self.project_path) / self.parameters.config_name
        config_path = str(config_path.resolve())

        with open(config_path, 'r') as fr:
            data = yaml.load(fr, Loader=yaml.FullLoader)
            bind_center_value = data['bind_center_value']
            b1_value = data['b1_value']
            b2_value = data['b2_value']

        for file in mat_files:
            file = str(Path(file).resolve())
            a = file[:file.find(f'_{encoder_type}_ego')]
            b = file.rsplit('_', 2)[1]

            # Remember to change this to either "True" or "False" depending on if you want to convert
            # the predicted Autoencoder points or ego_centered CNN points
            convert_auto = True
            if convert_auto:
                auto_cnn_name = f"{a}_{encoder_type}_ego_animal_{b}_data.mat"
            else:
                auto_cnn_name = f"{a}_{self.parameters.conv_tracker_name}_ego_animal_{b}_data.mat"

            ori_name = f"{a}_{self.parameters.conv_tracker_name}_animal_{b}_data.mat"

            if os.path.exists(ori_name) and os.path.exists(auto_cnn_name):
                if convert_auto:
                    filename = f"{a}_{encoder_type}_animal_{b}_data.mat"
                else:
                    filename = f"{a}_{self.parameters.conv_tracker_name}_r_animal_{b}_data.mat"

                if not os.path.exists(filename):
                    print(filename)
                    oriented_predicted = reorient(ori_name, auto_cnn_name, bind_center_value, b1_value, b2_value)
                    savemat(filename, {self.parameters.animal_key: oriented_predicted})

    def save_mat_to_h5(self, encoder_type='SAE'):

        mat_path = Path(self.project_path) / self.parameters.autoencoder_data_name
        mat_files = sorted(glob.glob(f'{str(mat_path)}/**/*{encoder_type}_animal*.mat', recursive=True))

        config_path = Path(self.project_path) / self.parameters.config_name
        config_path = str(config_path.resolve())

        with open(config_path, 'r') as fr:
            data = yaml.load(fr, Loader=yaml.FullLoader)
            body_parts = data['body_parts']

        for file in tqdm(mat_files):
            destination_path = Path(file).parents[0]
            file_s = Path(file).stem
            filename = destination_path / f'{file_s}.h5'
            if not filename.exists():
                print(filename.name)
                filename = str(filename.resolve())
                mat_file = loadmat(file)
                mat_data = mat_file[self.parameters.animal_key]
                index = np.arange(len(mat_data))

                if encoder_type == 'SAE':
                    scorer_name = 'Stacked_Autoencoder'
                elif encoder_type == 'VAE':
                    scorer_name = 'Variational_Autoencoder'

                iterables = [[scorer_name], body_parts, ['x', 'y']]
                # print(iterables)
                cols = pd.MultiIndex.from_product(iterables, names=['scorer', 'bodyparts', 'coords'])

                output_df = pd.DataFrame(mat_data, index=index, columns=cols)
                output_df.to_hdf(filename, self.parameters.animal_key)

    def combine_animal_h5_files(self, encoder_type='SAE'):

        h5_path = Path(self.project_path) / self.parameters.autoencoder_data_name
        h5_files = sorted(glob.glob(f'{str(h5_path)}/**/*{encoder_type}*animal*1*.h5', recursive=True))

        for file in h5_files:
            combine_h5_files(file)
