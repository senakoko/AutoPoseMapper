import os
from pathlib import Path
import cv2
from tqdm import tqdm


def down_sample_video(file, destination_path_path=None, scale_factor=0.5,
                      start_time=(1, 0), end_time=(2, 0), subset=True):
    """
    Reduces the resolution size of a video by a scale factor

    Parameters
    ----------
        file: video file
        destination_path_path: location to save down-sampled videos
        scale_factor: how much to scale down the video. From 0 to 1.
        subset: True or False - Create only a small subset of the video
        start_time: specifies the starting good point to use for the video
        end_time: specifies the ending good point to use for the video
    """

    vid_file = Path(file)

    # Destination filename
    sub_name = vid_file.stem
    destination_path_name = vid_file.name
    if destination_path_path is None:
        destination_path_path = str(vid_file.parents[0])
        destination_path_file = f"{destination_path_path}/DS_{sub_name}/DS_{destination_path_name}"
    else:
        destination_path_file = f"{destination_path_path}/DS_{sub_name}/DS_{destination_path_name}"

    if os.path.exists(destination_path_file):
        return
    else:
        if not os.path.exists(f"{destination_path_path}/DS_{sub_name}/"):
            os.makedirs(f"{destination_path_path}/DS_{sub_name}/")

    print(destination_path_file)

    # Video part
    cap = cv2.VideoCapture(file)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    height, width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(height * scale_factor)
    width = int(width * scale_factor)
    dim = (width, height)
    fps = cap.get(cv2.CAP_PROP_FPS)
    output = cv2.VideoWriter(destination_path_file, fourcc, fps, (width, height), 1)
    count = 0

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

    if subset:
        cap.set(1, start)
        for count in tqdm(range(int(start), int(end))):
            ret, image = cap.read()
            image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
            output.write(image)
    else:
        ret, image = cap.read()
        while ret:
            image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
            output.write(image)
            ret, image = cap.read()
            count += 1

    output.release()
    cap.release()
