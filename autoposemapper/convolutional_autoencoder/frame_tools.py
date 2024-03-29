import glob
import numpy as np
from pathlib import Path
from autoposemapper.convolutional_autoencoder.frame_extraction import extract_frames
import shutil
from autoposemapper.setRunParameters import set_run_parameter


class FrameTools:
    def __init__(self, project_path, parameters=None):
        self.project_path = project_path
        self.parameters = parameters

        if self.parameters is None:
            self.parameters = set_run_parameter()

    def extract_frames_conv(self, video_type='.mp4', destination_path=None, opencv=True,
                            numframes2pick=200, userfeedback=False, algo='uniform', name_prefix='extracted',
                            all_videos_or_subset='all', subset_num=10):

        video_path = Path(self.project_path) / self.parameters.conv_autoencoder_data_name
        video_files = sorted(glob.glob(f'{str(video_path)}/**/*{video_type}', recursive=True))

        if destination_path is None:
            destination_path = video_path

        if all_videos_or_subset == 'subset':
            idx = np.random.permutation(len(video_files))
            idx_slice = idx[:subset_num]

            videos = []
            for i in idx_slice:
                videos.append(video_files[i])
        else:
            videos = video_files

        for num, video in enumerate(videos):
            destination_folder = destination_path / f"{name_prefix}-data" / Path(video).stem
            destination_pictures = glob.glob(f"{destination_folder}/*.png")
            if len(destination_pictures) >= numframes2pick:
                print(f'Already extracted images from {Path(video).name}')
                continue
            print(f"Extracted {num}/{len(videos)}")
            print("Extracting ", Path(video).name)
            extract_frames(video, output_path=destination_path, numframes2pick=numframes2pick, opencv=opencv,
                           userfeedback=userfeedback, algo=algo, name_prefix=name_prefix)

    def create_train_test_datasets(self, train_fraction=0.8):

        frame_path = Path(self.project_path) / self.parameters.conv_autoencoder_data_name
        frame_files = sorted(glob.glob(f'{str(frame_path)}/**/*.png', recursive=True))

        frames2pick = np.arange(len(frame_files))
        train_size = int(len(frames2pick) * train_fraction)

        train_frames = np.random.choice(frames2pick, train_size, replace=False)
        test_frames = list(set(frames2pick) - set(train_frames))

        train_destination_path = Path(self.project_path) / self.parameters.conv_autoencoder_data_name / 'train/animals/'
        test_destination_path = Path(self.project_path) / self.parameters.conv_autoencoder_data_name / 'test/animals/'

        if not train_destination_path.exists():
            train_destination_path.mkdir(parents=True)

        if not test_destination_path.exists():
            test_destination_path.mkdir(parents=True)

        for fr in train_frames:
            file = frame_files[fr]
            destination_file = train_destination_path / f'{Path(file).parts[-2]}_{Path(file).name}'
            if not destination_file.exists():
                shutil.copy(file, destination_file)

        for fr in test_frames:
            file = frame_files[fr]
            destination_file = test_destination_path / f'{Path(file).parts[-2]}_{Path(file).name}'
            if not destination_file.exists():
                shutil.copy(file, destination_file)
