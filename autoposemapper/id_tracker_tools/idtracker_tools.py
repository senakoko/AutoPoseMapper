from pathlib import Path
import glob
import os
from tqdm import tqdm
import pandas as pd
from autoposemapper.id_tracker_tools.make_h5_id_traj import make_id_trajectories
from autoposemapper.id_tracker_tools.make_idt_movies import make_idt_movies
from autoposemapper.id_tracker_tools.createCSIcentroid import create_csi_cent
from autoposemapper.id_tracker_tools.createLabelsCSI import create_labels_csi
from autoposemapper.id_tracker_tools.swapClassifiedTracks import swap_classified_tracks
from autoposemapper.id_tracker_tools.fixBadArea import fix_bad_area
from autoposemapper.id_tracker_tools.createIndividualFromMulti import create_individuals_4_multi


class IdTracker:
    def __init__(self, project_path):
        self.project_path = project_path

    def make_h5s_id_trajectories(self):

        traj_path = Path(self.project_path) / 'id_tracker_data'
        traj_files = sorted(glob.glob(f"{traj_path}/**/*traj*.npy", recursive=True))

        for file in traj_files:
            make_id_trajectories(file)

    def make_id_tracker_movies(self, video_loc=None,
                               destination_path=None,
                               subset=False,
                               start_time=(1, 0),
                               end_time=(10, 0),
                               dot_size=10,
                               tracker='_idtraj'):

        h5_path = Path(self.project_path) / 'id_tracker_data'
        h5_files = sorted(glob.glob(f"{h5_path}/**/*idtraj.h5", recursive=True))

        for file in h5_files:
            make_idt_movies(file, video_loc, destination_path, subset, start_time,
                            end_time, dot_size, tracker)

    def create_csi_centroid(self, auto_source_path=None,
                            abs_tolerance=10,
                            dist_thresh=75,
                            area_thresh=1.1e4,
                            coord_points_path=None,
                            encoder_type='SAE'):

        h5_path = Path(self.project_path) / 'id_tracker_data'
        h5_files = sorted(glob.glob(f"{h5_path}/**/*idtraj.h5", recursive=True))

        for file in h5_files:
            create_csi_cent(file, auto_source_path=auto_source_path,
                            dist_thresh=dist_thresh, abs_tolerance=abs_tolerance,
                            area_thresh=area_thresh, encoder_type=encoder_type,
                            coord_points_path=coord_points_path)

    def create_labels_based_on_csi(self, destination_path=None, auto_source_path=None,
                                   abs_tolerance=40, tracker='_CSI', encoder_type='SAE',
                                   body_part1='betweenEars_midBody', body_part2='midBody_midHip'):

        h5_path = Path(self.project_path) / 'id_tracker_data'
        h5_files = sorted(glob.glob(f"{h5_path}/**/*centroid*.h5", recursive=True))

        for file in h5_files:
            create_labels_csi(file, destination_path=destination_path,
                              auto_source_path=auto_source_path,
                              abs_tolerance=abs_tolerance,
                              tracker=tracker, encoder_type=encoder_type,
                              body_part1=body_part1, body_part2=body_part2)

    def swap_tracks_based_on_labels(self, destination_path=None, auto_source_path=None,
                                    tracker='CSI_filtered',
                                    encoder_type='SAE'):

        h5_path = Path(self.project_path) / 'id_tracker_data'
        h5_files = sorted(glob.glob(f"{h5_path}/**/*labels*.h5", recursive=True))

        for file in h5_files:
            swap_classified_tracks(file, destination_path=destination_path,
                                   auto_source_path=auto_source_path,
                                   tracker=tracker, encoder_type=encoder_type)

    def fix_bad_areas_h5(self):

        h5_path = Path(self.project_path) / 'autoencoder_data'
        h5_files = sorted(glob.glob(f"{h5_path}/**/*CSI_filtered*.h5", recursive=True))

        for file in h5_files:
            fix_bad_area(file)

    def make_csi_filtered_not_present(self, encoder_type='SAE'):

        h5_path = Path(self.project_path) / 'autoencoder_data'
        h5_files = sorted(glob.glob(f"{h5_path}/**/*CNN_{encoder_type}*.h5", recursive=True))

        new_csi_files = []
        for file in h5_files:
            file_p = file.rsplit('_', 2)[0]
            csi_file = file_p + '_CSI_filtered.h5'
            if os.path.exists(csi_file):
                continue
            new_csi_files.append(file)

        for file in tqdm(new_csi_files):
            file_p = file.rsplit('_', 2)[0]
            destination_file = file_p + '_CSI_filtered.h5'
            if not os.path.exists(destination_file):
                h5 = pd.read_hdf(file)
                scorer = 'CSI_filtered'
                individuals = h5.columns.get_level_values('individuals').unique().to_list()
                bodyparts = h5.columns.get_level_values('bodyparts').unique().to_list()

                col = pd.MultiIndex.from_product([[scorer], individuals, bodyparts, ['x', 'y']],
                                                 names=['scorer', 'individuals', 'bodyparts', 'coords'])
                dataframe = pd.DataFrame(h5.values, index=h5.index, columns=col)
                print(destination_file)
                dataframe.to_hdf(destination_file, 'vole_d')

    def create_ind_from_multi(self, encoder_type='CSI'):

        h5_path = Path(self.project_path) / 'autoencoder_data'
        h5_files = sorted(glob.glob(f"{h5_path}/**/*_{encoder_type}*.h5", recursive=True))

        for file in h5_files:
            create_individuals_4_multi(file, tracker=encoder_type)
