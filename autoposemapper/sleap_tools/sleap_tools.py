import re
import warnings
import glob
import os
from pathlib import Path
import h5py
from autoposemapper.sleap_tools import utils
from autoposemapper.setRunParameters import set_run_parameter

warnings.filterwarnings('ignore')


class SleapHelper:
    def __init__(self, project_path, parameters=None):
        self.project_path = project_path
        self.parameters = parameters

        if self.parameters is None:
            self.parameters = set_run_parameter()

    def track_videos_sleap(self, model_path, video_type='.mp4'):
        """
        This is meant to track videos with SLEAP algorithm.
        It assumes that you already have trained SLEAP network that you are going to use on the videos.
        You will need to change the path to and the name of the network that you have trained.

        param
        model_path: string
            the path to trained model
        video_type: The video type can be an ".mp4" or ".avi"

        return:
        """

        video_path = Path(self.project_path) / self.parameters.sleap_data_name
        video_files = sorted(glob.glob(f'{str(video_path)}/**/*{video_type}', recursive=True))

        bash_file = Path(self.project_path) / self.parameters.bash_files_name / self.parameters.sleap_track_animal_name
        bash_file = bash_file.resolve()

        with open(bash_file, 'w') as f:
            print("#!/bin/bash", file=f)
            print("", file=f)
            gpu_to_use = input('Enter the gpu number to use: ')
            print(f'export CUDA_VISIBLE_DEVICES={gpu_to_use}', file=f)
            print('', file=f)

            for file in video_files:
                file = Path(file).resolve()
                nanim = input(f'How many animals are in {file.stem}: ')
                sleap_command = (f"sleap-track '{str(file)}' -m '{model_path}' "
                                 f"--tracking.clean_instance_count {nanim} --tracking.target_instance_count {nanim} "
                                 f"--tracking.tracker 'flow' --tracking.similarity 'instance' --batch_size 1"
                                 )
                print(sleap_command, file=f)
                print('', file=f)

        print("\n At this point, you can switch over to the terminal with SLEAP environment activated to run "
              "the bash scripts. I would suggest that because running the scripts through the jupyter notebook "
              "might cause the notebook to crush.\n")

    def clean_tracked_files(self):
        """
        clean SLEAP tracked data. It assumes that you already have tracked the data with SLEAP.
        You will need to change the path to and the name of tracked files.
        Most of the tracked files end with ".prediction.slp"
        return:

        """

        predictions_path = Path(self.project_path) / self.parameters.sleap_data_name
        predictions_files = sorted(glob.glob(f'{str(predictions_path)}/**/*predictions.slp', recursive=True))

        bash_file = Path(self.project_path) / self.parameters.bash_files_name / self.parameters.clean_track_animal_name
        bash_file = bash_file.resolve()

        with open(bash_file, 'w') as f:
            print("#!/bin/bash", file=f)
            print("", file=f)
            gpu_to_use = input('Enter the gpu number to use: ')
            print(f'export CUDA_VISIBLE_DEVICES={gpu_to_use}', file=f)
            print('', file=f)

            for file in predictions_files:
                file = Path(file).resolve()
                eventual_name = f'{str(file)[:-3]}cleaned.slp'
                if os.path.exists(eventual_name):
                    continue
                nanim = input(f'How many animals are in {file.stem}: ')
                sleap_command = f"python -m sleap.info.trackcleaner '{str(file)}' -c {nanim} "
                print(sleap_command, file=f)
                print('', file=f)

        print("\nAt this point, you can switch over to the terminal with SLEAP environment "
              "activated to run the bash scripts. I would suggest that because running "
              "the scripts through the jupyter notebook might cause the notebook to crush.\n")

        print("\n After you have created the '.cleaned.slp' file, you would have to open those "
              "files with the SLEAP GUI to check the tracking. You would want the instances "
              "connected to the individual that they are supposed to.\n")

    def convert_sleap_2_h5(self):
        """
        convert SLEAP tracked data to h5 file format. It assumes that you already have cleaned the data with SLEAP.
        You will need to change the path to and the name of cleaned files.
        Most of the tracked files end with ".predictions.cleaned.slp"
        return:
        """

        predictions_path = Path(self.project_path) / self.parameters.sleap_data_name
        predictions_files = sorted(glob.glob(f'{str(predictions_path)}/**/*predictions.cleaned.slp', recursive=True))

        bash_file = Path(self.project_path) / self.parameters.bash_files_name / self.parameters.convert_cleaned_slp
        bash_file = bash_file.resolve()

        with open(bash_file, 'w') as f:
            print("#!/bin/bash", file=f)
            print("", file=f)
            gpu_to_use = input('Enter the gpu number to use: ')
            print(f'export CUDA_VISIBLE_DEVICES={gpu_to_use}', file=f)
            print('', file=f)

            for file in predictions_files:
                file = Path(file).resolve()
                file_s = str(file)
                eventual_name = f"{file_s[:file_s.find('.predictions.cleaned.slp') - 4]}.h5"
                if os.path.exists(eventual_name):
                    continue
                sleap_command = f"sleap-convert '{file_s}' -o '{eventual_name}' --format analysis "
                print(sleap_command, file=f)
                print('', file=f)

        print("\n At this point, you can switch over to the terminal with "
              "SLEAP environment activated to run the bash scripts. "
              "I would suggest that because running the scripts through "
              "the jupyter notebook might cause the notebook to crush.\n")

    def check_sleap_converted_h5(self):

        h5_path = Path(self.project_path) / self.parameters.sleap_data_name
        h5_files = sorted(glob.glob(f'{str(h5_path)}/**/*.h5', recursive=True))

        for file in h5_files:

            with h5py.File(file, "r") as f:
                dset_names = list(f.keys())
                locations = f["tracks"][:].T
                node_names = [n.decode() for n in f["node_names"][:]]

            print("===filename===")
            print(Path(file).stem)
            print()

            print("===HDF5 datasets===")
            print(dset_names)
            print()

            print("===locations data shape===")
            print(locations.shape)
            print()

            print("===nodes===")
            for i, name in enumerate(node_names):
                print(f"{i}: {name}")
            print()

    def convert_sleap_h5_2_pandas_h5(self):
        """
        convert Sleap's 'h5' to Pandas style/ DLC 'h5' format
        """
        h5_path = Path(self.project_path) / self.parameters.sleap_data_name
        h5_files = sorted(glob.glob(f'{str(h5_path)}/**/*.h5', recursive=True))

        destination_path = Path(self.project_path) / self.parameters.autoencoder_data_name

        sleap_files = []
        # Check if the file has already been processed
        patterns = ['CNN']
        for file in h5_files:
            pattern_value = False
            for pattern in patterns:
                if re.search(pattern, file):
                    pattern_value = True
                    break
            if pattern_value:
                continue
            else:
                sleap_files.append(file)

        for file in sleap_files:
            utils.convert_sh5_to_ph5(file, destination_path=destination_path)
