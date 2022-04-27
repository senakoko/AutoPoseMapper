import glob
import numpy as np
from pathlib import Path
from autoposemapper.convolutional_autoencoder.frame_extraction import extract_frames
import shutil


class FrameTools:
    def __init__(self, project_path):
        self.project_path = project_path

    def extract_frames_conv(self, video_type='.mp4', destination_path=None, opencv=True,
                            numframes2pick=200, userfeedback=False, algo='uniform', name_prefix='extracted'):

        video_path = Path(self.project_path) / 'conv_autoencoder_data'
        video_files = sorted(glob.glob(f'{str(video_path)}/**/*{video_type}', recursive=True))

        if destination_path is None:
            destination_path = video_path

        idx = np.random.permutation(len(video_files))
        idx_slice = idx[:30]

        videos = []
        for i in idx_slice:
            videos.append(video_files[i])

        for video in videos:
            extract_frames(video, output_path=destination_path, numframes2pick=numframes2pick, opencv=opencv,
                           userfeedback=userfeedback, algo=algo, name_prefix=name_prefix)

    def create_train_test_datasets(self, train_fraction=0.8):

        frame_path = Path(self.project_path) / 'conv_autoencoder_data'
        frame_files = sorted(glob.glob(f'{str(frame_path)}/**/*.png', recursive=True))

        frames2pick = np.arange(len(frame_files))
        train_size = int(len(frames2pick) * train_fraction)

        train_frames = np.random.choice(frames2pick, train_size, replace=False)
        test_frames = list(set(frames2pick) - set(train_frames))

        train_destination_path = Path(self.project_path) / 'conv_autoencoder_data' / 'train/animals/'
        test_destination_path = Path(self.project_path) / 'conv_autoencoder_data' / 'test/animals/'

        if not train_destination_path.exists():
            train_destination_path.mkdir(parents=True)

        if not test_destination_path.exists():
            test_destination_path.mkdir(parents=True)

        for fr in train_frames:
            file = frame_files[fr]
            destination_file = train_destination_path / Path(file).name
            if not destination_file.exists():
                shutil.copy(file, destination_file)

        for fr in test_frames:
            file = frame_files[fr]
            destination_file = test_destination_path / Path(file).name
            if not destination_file.exists():
                shutil.copy(file, destination_file)
