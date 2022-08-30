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

        files = sorted(glob.glob(f'{str(dlc_path)}*DLC*', recursive=True))
        destination_path = Path(self.project_path) / self.parameters.dlc_data_name

        for file in files:
            destination_file = f'{destination_path}/{Path(file).name}'
            shutil.copy(file, destination_file)

    def copy_dlc_files_AE(self):
        """
        copy DLC h5 files to the autoencoder data path
        """

        h5_path = Path(self.project_path) / self.parameters.dlc_data_name
        h5_files = sorted(glob.glob(f'{str(h5_path)}/*.h5'))

        destination_path = Path(self.project_path) / self.parameters.autoencoder_data_name

        for file in h5_files:
            file_s = Path(file).stem
            file_s = file_s[:file_s.find('DLC')]
            destination_file = f'{destination_path}/{file_s}_{self.parameters.conv_tracker_name}.h5'
            shutil.copy(file, destination_file)
