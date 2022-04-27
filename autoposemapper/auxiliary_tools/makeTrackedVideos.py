import os
import glob
from pathlib import Path
import pandas as pd
import cv2
from skimage import draw
import numpy as np
from tqdm import tqdm
import yaml


def create_body_indices(bodyparts, skeleton):
    bpts_val = {}
    for i, bpts in enumerate(bodyparts):
        bpts_val[bpts] = i
    sk_num = []
    for sk in skeleton:
        sk_val = [bpts_val[sk[0]], bpts_val[sk[1]]]
        sk_num.append(sk_val)
    return sk_num


def make_tracked_movies(tracked_file, video_loc=None, skeleton_path=None, 
                        destination_path=None, subset=False, 
                        start_time=(1, 0), end_time=(2, 0),
                        post_name='CNN_SAE', dot_size=4, tracker='CNN_SAE',
                        no_tracker=False):
    """
    Create videos with ID tracker tracking.

    Parameters:
    -----------
        tracked_file: the h5 file with the IDT tracking.
        video_loc: path to the video
        skeleton_path: path to the skeleton to plot on videos.
        destination_path: The folder path to save the videos.
        subset: True or False - Create only a small subset of the video
        start_time: specifies the starting good point to use for the video
        end_time: specifies the ending good point to use for the video
        post_name: post_name added to movie file
        dot_size: size of tracked point

    Output:
    -------
        movie: saved video with IDT tracking
    """

    file = Path(tracked_file)
    if no_tracker:
        file_s = file.stem
    else:
        file_s = file.stem[:file.stem.find(f'_{tracker}')]
    file_vid = file_s + '.mp4'

    if skeleton_path is None:
        print('Include the path to the skeleton.yaml file to create tracked movies')
        return
    else:
        with open(skeleton_path, 'r') as file:
            skeleton = yaml.safe_load(file)
        skeleton = skeleton['skeleton']

    if os.path.isfile(video_loc):
        vid_file = Path(video_loc)
    else:
        video_path = glob.glob(f'{video_loc}/**/{file_vid}', recursive=True)
        print(video_path)
        video_loc = video_path[0]
        vid_file = Path(video_loc)

    # Check if the idtracker h5 file and video name match.
    # If they do then run analysis

    if file_vid in vid_file.name:

        if destination_path is None:
            destination_path = str(Path(video_loc).parents[0])
            if no_tracker:
                file_loc = f"{destination_path}/{file_s}_original.mp4"
            else:
                file_loc = f"{destination_path}/{file_s}_{post_name}.mp4"
        else:
            if no_tracker:
                file_loc = f"{destination_path}/{file_s}_original.mp4"
            else:
                file_loc = f"{destination_path}/{file_s}_{post_name}.mp4"

        if os.path.exists(file_loc):
            return

        print(file_loc)

        # Color
        # Red for individual 1 and Blue for 2
        color = [[0, 0, 1], [(1, 0, 0)]]

        # h5 file part
        h5 = pd.read_hdf(tracked_file)
        scorer = h5.columns.get_level_values('scorer').unique()
        scorer = scorer.item()
        bodyparts = h5.columns.get_level_values('bodyparts').unique().to_list()
        individuals = h5.columns.get_level_values('individuals').unique().to_list()

        # Video part
        cap = cv2.VideoCapture(video_loc)
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        height, width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fps = cap.get(cv2.CAP_PROP_FPS)
        output = cv2.VideoWriter(file_loc, fourcc, fps, (width, height), 1)
        ny, nx = height, width

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

        bpt_indices = create_body_indices(bodyparts, skeleton)

        if not subset:
            cap.set(1, 0)
            for count in tqdm(range(h5.shape[0])):
                ret, image = cap.read()
                for j, ind in enumerate(individuals):

                    individual = h5[scorer][ind]
                    df_x, df_y = individual.values.reshape((len(individual), -1, 2)).T

                    for bp1, bp2 in bpt_indices:

                        if not (np.any(np.isnan(df_x[[bp1, bp2], count]))
                                or np.any(np.isnan(df_y[[bp1, bp2], count]))):
                            rr, cc, val = draw.line_aa(
                                int(np.clip(df_y[bp1, count], 0, ny - 1)),
                                int(np.clip(df_x[bp1, count], 0, nx - 1)),
                                int(np.clip(df_y[bp2, count], 0, ny - 1)),
                                int(np.clip(df_x[bp2, count], 0, nx - 1))
                            )
                            image[rr, cc] = (np.array([1, 1, 1]) * 255).astype(np.uint8)

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

                    for bp1, bp2 in bpt_indices:
                        if not (np.any(np.isnan(df_x[[bp1, bp2], count]))
                                or np.any(np.isnan(df_y[[bp1, bp2], count]))):
                            rr, cc, val = draw.line_aa(
                                int(np.clip(df_y[bp1, count], 0, ny - 1)),
                                int(np.clip(df_x[bp1, count], 0, nx - 1)),
                                int(np.clip(df_y[bp2, count], 0, ny - 1)),
                                int(np.clip(df_x[bp2, count], 0, nx - 1))
                            )
                            image[rr, cc] = (np.array([1, 1, 1]) * 255).astype(np.uint8)

                    for i, bp in enumerate(bodyparts):
                        rr, cc = draw.disk((df_y[i, count], df_x[i, count]), dot_size, shape=image.shape)
                        image[rr, cc, :] = (np.array(color[j]) * 255).astype(np.uint8)

                output.write(image)

        output.release()
        cap.release()
        print(f'Done creating video for {file_s}.mp4')
    else:
        print(f'Could not find a video match to the IDT file')


def make_tracked_movies_idt(tracked_file, video_loc=None,
                            skeleton_path=None, destination_path=None, subset=False,
                            start_time=(1, 0),
                            end_time=(2, 0), includeIDT=True,
                            pathtoIDT=None, post_name='CNN_SAE_IDT',
                            dot_size=4, tracker='CNN',
                            dot_size_idt=10):
    """
    Create videos with ID tracker tracking

    Parameters:
    -----------
        file: the h5 file with the IDT tracking
        video_loc: path to the video
        skeleton_path: path to the skeleton to plot on videos.
        destination_path: The folder path to save the videos.
        subset: True or False - Create only a small subset of the video
        start_time: specifies the starting good point to use for the video
        end_time: specifies the ending good point to use for the video
        includeIDT: Boolean, True or False.- Include ID Tracker tracking
        pathtoIDT: the path to ID Tracker tracking
        post_name: post_name added to movie file
        dot_size: size of tracked point

    Output:
    -------
        movie: saved video with IDT tracking
    """

    file = Path(tracked_file)
    file_s = file.stem[:file.stem.find(f'_{tracker}')]
    file_vid = file_s + '.mp4'

    # Skeleton path
    if skeleton_path is None:
        print('Include the path to the skeleton.yaml file to create tracked movies')
        return
    else:
        with open(skeleton_path, 'r') as file:
            skeleton = yaml.safe_load(file)
        skeleton = skeleton['skeleton']

    # Video file path
    if os.path.isfile(video_loc):
        vid_file = Path(video_loc)
    else:
        video_path = glob.glob(f'{video_loc}/**/{file_vid}', recursive=True)
        video_loc = video_path[0]
        vid_file = Path(video_loc)
        print(vid_file)

    # ID Tracker file
    if includeIDT:
        if os.path.isfile(pathtoIDT):
            idt_file = Path(pathtoIDT)
        else:
            idt_path = glob.glob(f'{pathtoIDT}/**/{file_s}*_idtraj*h5', recursive=True)
            idt_loc = idt_path[0]
            idt_file = Path(idt_loc)
            print(idt_file)
    else:
        print('Include the path to ID tracking file')
        return

    # Check if the idtracker h5 file and video name match.
    # If they do then run analysis

    if file_vid in vid_file.name:

        if destination_path is None:
            destination_path = str(Path(video_loc).parents[0])
            file_loc = f"{destination_path}/{file_s}_{post_name}.mp4"
        else:
            file_loc = f"{destination_path}/{file_s}_{post_name}.mp4"

        if os.path.exists(file_loc):
            return

        print(file_loc)

        # Color
        # Red for individual 1 and Blue for 2
        color = [[0, 0, 1], [(1, 0, 0)]]
        color_idt = [[0.5, 0.5, 1], [(1, 0.5, 0.5)]]

        # h5 file part
        h5 = pd.read_hdf(tracked_file)
        scorer = h5.columns.get_level_values('scorer').unique().item()
        bodyparts = h5.columns.get_level_values('bodyparts').unique().to_list()
        individuals = h5.columns.get_level_values('individuals').unique().to_list()

        # IDT file part
        h5_idt = pd.read_hdf(idt_file)
        scorer_idt = h5_idt.columns.get_level_values('scorer').unique().item()
        bodyparts_idt = h5_idt.columns.get_level_values('bodyparts').unique().to_list()

        # Video part
        cap = cv2.VideoCapture(video_loc)
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        height, width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fps = cap.get(cv2.CAP_PROP_FPS)
        output = cv2.VideoWriter(file_loc, fourcc, fps, (width, height), 1)
        ny, nx = height, width

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

        bpt_indices = create_body_indices(bodyparts, skeleton)

        if not subset:
            cap.set(1, 0)
            for count in tqdm(range(h5.shape[0])):
                ret, image = cap.read()
                for j, ind in enumerate(individuals):

                    individual = h5[scorer][ind]
                    df_x, df_y = individual.values.reshape((len(individual), -1, 2)).T

                    individual_idt = h5_idt[scorer_idt][ind]
                    df_x_idt, df_y_idt = individual_idt.values.reshape((len(individual_idt), -1, 2)).T

                    for bp1, bp2 in bpt_indices:
                        if not (np.any(np.isnan(df_x[[bp1, bp2], count]))
                                or np.any(np.isnan(df_y[[bp1, bp2], count]))):
                            rr, cc, val = draw.line_aa(
                                int(np.clip(df_y[bp1, count], 0, ny - 1)),
                                int(np.clip(df_x[bp1, count], 0, nx - 1)),
                                int(np.clip(df_y[bp2, count], 0, ny - 1)),
                                int(np.clip(df_x[bp2, count], 0, nx - 1))
                            )
                            image[rr, cc] = (np.array([1, 1, 1]) * 255).astype(np.uint8)

                    for i, bp in enumerate(bodyparts):
                        rr, cc = draw.disk((df_y[i, count], df_x[i, count]), dot_size, shape=image.shape)
                        image[rr, cc, :] = (np.array(color[j]) * 255).astype(np.uint8)

                    for i, bp in enumerate(bodyparts_idt):
                        rr, cc = draw.disk((df_y_idt[i, count], df_x_idt[i, count]), dot_size, shape=image.shape)
                        image[rr, cc, :] = (np.array(color[j]) * 255).astype(np.uint8)

                output.write(image)
        else:
            cap.set(1, start)
            for count in tqdm(range(int(start), int(end))):
                ret, image = cap.read()
                for j, ind in enumerate(individuals):

                    individual = h5[scorer][ind]
                    df_x, df_y = individual.values.reshape((len(individual), -1, 2)).T

                    individual_idt = h5_idt[scorer_idt][ind]
                    df_x_idt, df_y_idt = individual_idt.values.reshape((len(individual_idt), -1, 2)).T

                    for bp1, bp2 in bpt_indices:
                        if not (np.any(np.isnan(df_x[[bp1, bp2], count]))
                                or np.any(np.isnan(df_y[[bp1, bp2], count]))):
                            rr, cc, val = draw.line_aa(
                                int(np.clip(df_y[bp1, count], 0, ny - 1)),
                                int(np.clip(df_x[bp1, count], 0, nx - 1)),
                                int(np.clip(df_y[bp2, count], 0, ny - 1)),
                                int(np.clip(df_x[bp2, count], 0, nx - 1))
                            )
                            image[rr, cc] = (np.array([1, 1, 1]) * 255).astype(np.uint8)

                    for i, bp in enumerate(bodyparts):
                        rr, cc = draw.disk((df_y[i, count], df_x[i, count]), dot_size, shape=image.shape)
                        image[rr, cc, :] = (np.array(color[j]) * 255).astype(np.uint8)

                    for i, bp in enumerate(bodyparts_idt):
                        rr, cc = draw.disk((df_y_idt[i, count], df_x_idt[i, count]), dot_size_idt, shape=image.shape)
                        image[rr, cc, :] = (np.array(color_idt[j]) * 255).astype(np.uint8)

                output.write(image)

        output.release()
        cap.release()
        print(f'Done creating video for {file_s}.mp4')
    else:
        print(f'Could not find a video match to the IDT file')
        