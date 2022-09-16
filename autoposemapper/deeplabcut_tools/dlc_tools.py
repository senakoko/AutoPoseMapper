import os
import shutil
import warnings
import glob
from pathlib import Path
from autoposemapper.setRunParameters import set_run_parameter


warnings.filterwarnings('ignore')


class DlcHelper:
    def __init__(self, project_path, parameters=None):
        self.project_path = project_path
        self.parameters = parameters

        if self.parameters is None:
            self.parameters = set_run_parameter()

    def copy_dlc_files_DD(self, dlc_path):
        """
        copy DLC files to the deeplabcut data path
        """

        files = sorted(glob.glob(f'{str(dlc_path)}/**/*DLC*', recursive=True))
        destination_path = Path(self.project_path) / self.parameters.dlc_data_name

        for file in files:
            destination_file = f'{destination_path}/{Path(file).name}'
            shutil.copy(file, destination_file)

    def copy_dlc_files_AE(self, dlc_data_path=None):
        """
        copy DLC h5 files to the autoencoder data path
        """

        if dlc_data_path is None:
            dlc_data_path = Path(self.project_path) / self.parameters.dlc_data_name
        h5_files = sorted(glob.glob(f'{str(dlc_data_path)}/**/*DLC*.h5', recursive=True))

        destination_path = Path(self.project_path) / self.parameters.autoencoder_data_name

        for file in h5_files:
            file_s = Path(file).stem
            file_s = file_s[:file_s.find('DLC')]
            destination_file = f'{destination_path}/{file_s}_{self.parameters.conv_tracker_name}.h5'
            shutil.copy(file, destination_file)

    def copy_labeled_DLC_h5(self, labeled_data_path=None):
        """
        Skip this function if you don't have the manually labeled data used to train the DLC network.
        Copy the labeled h5 files to the autoencoder data path. If you have this data, you will use it to train your
        autoencoder network the first time.
        Parameters
        ----------
        labeled_data_path: Path to the data used to train the DLC network

        Returns
        -------
        """

        if labeled_data_path is None:
            print("Please provide the path to DLC labeled-data. Skip this function if you don't have the data")
            return

        h5_files = sorted(glob.glob(f'{labeled_data_path}/**/Coll*.h5', recursive=True))

        destination_path = Path(self.project_path) / self.parameters.autoencoder_data_name

        for file in h5_files:
            file_s = Path(file).stem
            file_p = Path(file).parts
            sub_folder = '/'.join(file_p[-3:-1])
            sub_folder_path = f'{destination_path}/{sub_folder}'
            if not os.path.exists(sub_folder_path):
                os.makedirs(sub_folder_path, exist_ok=True)
            destination_file = f'{sub_folder_path}/{file_s}_{self.parameters.conv_tracker_name}.h5'
            shutil.copy(file, destination_file)
