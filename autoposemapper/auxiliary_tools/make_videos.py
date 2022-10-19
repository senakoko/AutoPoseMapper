from pathlib import Path
import glob
import re

from autoposemapper.auxiliary_tools.makeTrackedVideos import make_tracked_movies, make_tracked_movies_idt
from autoposemapper.setRunParameters import set_run_parameter


class MakeVideos:
    def __init__(self, project_path, parameters=None):
        self.project_path = project_path

        if parameters is None:
            self.parameters = set_run_parameter()

    def make_tracked_videos(self, video_loc=None, skeleton_path=None,
                            destination_path=None, subset=False,
                            start_time=(1, 0), end_time=(10, 0),
                            post_name='CNN_SAE', dot_size=4, tracker='CNN_SAE',
                            no_tracker=False):

        h5_path = Path(self.project_path) / self.parameters.autoencoder_data_name
        h5_files = sorted(glob.glob(f"{h5_path}/**/*{tracker}*.h5", recursive=True))

        files = []
        for file in h5_files:
            if re.search('labeled', file):
                continue
            files.append(file)

        for i, file in enumerate(files):
            make_tracked_movies(file, video_loc=video_loc, skeleton_path=skeleton_path,
                                destination_path=destination_path, subset=subset,
                                start_time=start_time, end_time=end_time,
                                post_name=post_name, dot_size=dot_size, tracker=tracker,
                                no_tracker=no_tracker)
            if i == 2:
                break

    def make_tracked_videos_w_idt(self, video_loc=None, skeleton_path=None,
                                  destination_path=None, subset=False,
                                  start_time=(1, 0), end_time=(10, 0),
                                  post_name='CNN_SAE', dot_size=4, tracker='CNN_SAE'):

        h5_path = Path(self.project_path) / self.parameters.autoencoder_data_name
        h5_files = sorted(glob.glob(f"{h5_path}/**/*{tracker}*.h5", recursive=True))

        idt_path = Path(self.project_path) / self.parameters.id_tracker_data_name

        files = []
        for file in h5_files:
            if re.search('labeled', file):
                continue
            files.append(file)

        for file in h5_files:
            make_tracked_movies_idt(file, video_loc=video_loc, skeleton_path=skeleton_path,
                                    destination_path=destination_path, subset=subset,
                                    start_time=start_time, end_time=end_time,
                                    post_name=post_name, dot_size=dot_size, pathtoIDT=idt_path)
