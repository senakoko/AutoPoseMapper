import glob
from pathlib import Path
from autoposemapper.motion_mapper_tools.calBodypartDistances import cal_bodypart_distances


class CalculateBodyInfo:
    def __init__(self, project_path):
        self.project_path = project_path

    def calculate_body_info(self, calculation_type='Euc_Dist', encoder_type='SAE'):

        h5_path = Path(self.project_path) / 'autoencoder_data'
        h5_files = sorted(glob.glob(f'{str(h5_path)}/**/*{encoder_type}_animal_*.h5'))

        destination_path = Path(self.project_path) / f'{encoder_type}_{calculation_type}'
        destination_path = destination_path.resolve()

        if not destination_path.exists():
            destination_path.mkdir(parents=True)
            print(f'{destination_path.stem} folder made')

        for file in h5_files:
            if calculation_type == 'Euc_Dist':
                cal_bodypart_distances(file, destination_path, encoder_type=encoder_type)

        return destination_path
