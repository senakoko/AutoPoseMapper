import glob
from pathlib import Path
from autoposemapper.motion_mapper_tools.calBodypartDistances import cal_bodypart_distances_multi
from autoposemapper.motion_mapper_tools.calJointAngles import cal_joint_angles
from autoposemapper.setRunParameters import set_run_parameter


class CalculateBodyInfo:
    def __init__(self, project_path, parameters=None):
        self.project_path = project_path
        self.parameters = parameters

        if self.parameters is None:
            self.parameters = set_run_parameter()

    def calculate_body_info(self, calculation_type='Euc_Dist', encoder_type='CNN'):

        h5_path = Path(self.project_path) / self.parameters.autoencoder_data_name
        h5_files = sorted(glob.glob(f'{str(h5_path)}/**/*{encoder_type}*_filtered.h5', recursive=True))

        destination_path = Path(self.project_path) / f'{calculation_type}'
        destination_path = destination_path.resolve()

        if not destination_path.exists():
            destination_path.mkdir(parents=True)
            print(f'{destination_path.stem} folder made')

        for file in h5_files:
            if calculation_type == 'Euc_Dist':
                cal_bodypart_distances_multi(file, destination_path, encoder_type=encoder_type)
            elif calculation_type == 'Joint_Angles':
                cal_joint_angles(file, destination_path, encoder_type=encoder_type)

        return destination_path
