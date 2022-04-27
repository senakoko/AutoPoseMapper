import os
import glob
from pathlib import Path
import pandas as pd
import cv2
from skimage import draw
import numpy as np
from tqdm import tqdm


def make_idt_movies(file,
                    video_loc=None,
                    destination_path=None,
                    subset=False,
                    start_time=(1, 0),
                    end_time=(10, 0),
                    dot_size=10,
                    tracker='_idtraj'):
    """
    Create videos with ID tracker tracking

    Parameters:
    -----------
        file: the h5 file with the IDT tracking
        video_loc: path to the video
        destination_path: The folder path to save the videos.
        subset: True or False - Create only a small subset of the video
        start_time: specifies the starting good point to use for the video
        end_time: specifies the ending good point to use for the video
        dot_size: size of tracked point

    Output:
    -------
        movie: saved video with IDT tracking
    """

    file = Path(file)
    file_s = file.stem[:file.stem.find(tracker)]
    file_vid = file_s + '.mp4'

    if os.path.isfile(video_loc):
        vid_file = Path(video_loc)
    else:
        video_path = glob.glob(f'{video_loc}/**/{file_vid}', recursive=True)
        video_loc = video_path[0]
        vid_file = Path(video_loc)
        print(vid_file)

    # Check if the idtracker h5 file and video name match.
    # If they do then run analysis

    if file_vid in vid_file.name:

        if destination_path is None:
            destination_path = str(Path(video_loc).parents[0].resolve())
            file_loc = f"{destination_path}/{file.stem}.mp4"
        else:
            file_loc = f"{destination_path}/{file.stem}.mp4"

        if os.path.exists(file_loc):
            return

        print(file_loc)

        # Color
        # Red for individual 1 and Blue for 2
        color = [[0, 0, 1], [(0, 1, 0)]]

        # h5 file part
        h5 = pd.read_hdf(file)
        scorer = h5.columns.get_level_values('scorer').unique().item()
        bodyparts = h5.columns.get_level_values('bodyparts').unique().to_list()
        individuals = h5.columns.get_level_values('individuals').unique().to_list()

        # Video part
        cap = cv2.VideoCapture(video_loc)
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        height, width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fps = cap.get(cv2.CAP_PROP_FPS)
        output = cv2.VideoWriter(file_loc, fourcc, fps, (width, height), 1)

        # Determine start_time and end_time
        second = 60
        minute = 60
        if type(start_time) == int:
            start_time = (start_time,)
        if len(start_time) == 1:
            start = start_time[0] * fps
        elif len(start_time) == 2:
            start = (start_time[0] * second * fps) + (start_time[1] * fps)
        elif len(start_time) == 3:
            start = (start_time[0] * minute * second * fps) + (start_time[1] * second * fps) + (start_time[2] * fps)

        if type(end_time) == int:
            end_time = (end_time,)
        if len(end_time) == 1:
            end = end_time[0] * fps
        elif len(end_time) == 2:
            end = (end_time[0] * second * fps) + (end_time[1] * fps)
        elif len(end_time) == 3:
            end = (end_time[0] * minute * second * fps) + (end_time[1] * second * fps) + (end_time[2] * fps)

        assert end > start

        if not subset:
            cap.set(1, 0)
            for count in tqdm(range(h5.shape[0])):
                ret, image = cap.read()
                for j, ind in enumerate(individuals):

                    individual = h5[scorer][ind]
                    df_x, df_y = individual.values.reshape((len(individual), -1, 2)).T

                    for i, bp in enumerate(bodyparts):
                        rr, cc = draw.disk((df_y[i, count], df_x[i, count]), dot_size, shape=image.shape)
                        image[rr, cc, :] = (np.array(color[j]) * 255).astype(np.uint8)

                output.write(image)
        else:
            cap.set(1, start)
            for count in tqdm(range(int(start), int(end))):
                ret, image = cap.read()
                for j, ind in enumerate(individuals):

                    individual = h5[scorer][ind]
                    df_x, df_y = individual.values.reshape((len(individual), -1, 2)).T

                    for i, bp in enumerate(bodyparts):
                        rr, cc = draw.disk((df_y[i, count], df_x[i, count]), dot_size, shape=image.shape)
                        image[rr, cc, :] = (np.array(color[j]) * 255).astype(np.uint8)

                output.write(image)

        output.release()
        cap.release()

        print(f'Done creating video for {file.stem}.mp4')
    else:
        print(f'Could not find a video match to the IDT file')
