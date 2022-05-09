import glob
import hdf5storage
import numpy as np
from moviepy.editor import *
from pathlib import Path


def make_brady_movie_parallel(reg, destination_path=None, groups=None, clips=None):
    """
    Create mosaic movies alias brady movies as a parallel stack.
    reg: region number from map
    destination_path: destination path to save videos
    groups: groups data from map
    clips: the video clips
    """

    vid_dim = 6
    region_on_map = reg + 1
    selected_clip = groups.item(reg)
    if not selected_clip.shape[0]:
        return
    comp_clips = []
    vid_end_min = []
    for sc in selected_clip:
        vid_end = sc[2] - sc[1]
        vid_end_min.append(vid_end)
    vid_e = np.median(vid_end_min)
    for sc in selected_clip:
        vid_end = sc[2] - sc[1]
        if vid_end >= vid_e:
            vid_index, vid_start = sc[0], round((sc[1] / 30), 2)
            vid_end = round((sc[1] + vid_e) / 30, 2)
            # print(f"{vid_index}-{vid_start}-{vid_end}")
            reg_vid = clips[vid_index - 1].subclip(vid_start, vid_end)
            # Generate a text clip  
            txt_clip = TextClip(f"{vid_index}", fontsize=20, color='red',
                                bg_color='White', stroke_width=1.5)
            txt_clip = txt_clip.set_pos(('right', 'top')).set_duration(reg_vid.duration)
            # Overlay the text clip on the video clip
            video = CompositeVideoClip([reg_vid, txt_clip])
            comp_clips.append(video)
            for i in range(10):
                np.random.shuffle(comp_clips)
        else:
            continue
    clip_vids = []
    vids = []
    for ind, i in enumerate(comp_clips):
        if len(vids) == vid_dim:
            clip_vids.append(vids)
            # vids = []
            vids.append(i)
        else:
            vids.append(i)
        if ind + 1 == len(comp_clips):
            if len(vids) != vid_dim:
                extra = vid_dim - len(vids)
                vids.extend(comp_clips[:extra])
            clip_vids.append(vids)
    clip_vids = clip_vids[:4]
    final_clip = clips_array(clip_vids)
    final_clip.write_videofile(f"{destination_path}reg_{region_on_map}_vid.mp4")


def make_brady_movie_series(reg, destination_path=None, groups=None, clips=None):
    """
    Create mosaic movies alias brady movies as a series stack.
    reg: region number from map
    destination_path: destination path to save videos
    groups: groups data from map
    clips: the video clips
    """

    region_on_map = reg + 1
    selected_clip = groups.item(reg)
    if not selected_clip.shape[0]:
        return
    comp_clips = []
    vid_end_min = []
    for sc in selected_clip:
        vid_end = sc[2] - sc[1]
        vid_end_min.append(vid_end)
    vid_e = np.median(vid_end_min)
    for sc in selected_clip:
        vid_end = sc[2] - sc[1]
        if vid_end >= vid_e:
            vid_index, vid_start = sc[0], round((sc[1] / 30), 2)
            vid_end = round((sc[1] + vid_e) / 30, 2)
            # print(f"{vid_index}-{vid_start}-{vid_end}")
            reg_vid = clips[vid_index - 1].subclip(vid_start, vid_end)
            # Generate a text clip  
            txt_clip = TextClip(f"{vid_index}", fontsize=20, color='red',
                                bg_color='White', stroke_width=1.5)
            txt_clip = txt_clip.set_pos(('right', 'top')).set_duration(reg_vid.duration)
            # Overlay the text clip on the video clip
            video = CompositeVideoClip([reg_vid, txt_clip])
            comp_clips.append(video)
            for i in range(10):
                np.random.shuffle(comp_clips)
        else:
            continue
    final_clip = concatenate_videoclips(comp_clips)
    final_clip.write_videofile(f"{destination_path}reg_{region_on_map}_vid.mp4")


def create_brady_videos(watershed_path, video_path, make_video_parallel_or_series='parallel'):
    gp_data = str(Path(watershed_path).resolve())
    groups_data = hdf5storage.loadmat(gp_data)
    groups = abs(groups_data['groups'])
    zValNames = groups_data['zValNames'][0]

    vid_file_path = []
    for i in zValNames:
        file = i.item(0)
        file = file[:file.find('_pca')] + '.mp4'
        vid_files = f"{video_path}{file}"
        vid_file_path.append(vid_files)

    clips = [VideoFileClip(vid_name) for vid_name in vid_file_path]

    output_dir = Path(watershed_path).parents[0]
    output_dir = output_dir.parents[0]
    output_dir = output_dir / 'Brady_Movies1'
    if output_dir.exists():
        output_dirs = glob.glob(f'{str(output_dir.resolve())[:-1]}*')[-1]
        numb = int(Path(output_dirs).stem[-1])
        output_dir = output_dir.parents[0] / f'Brady_Movies{numb + 1}'
        output_dir.mkdir(parents=True)
    else:
        output_dir.mkdir(parents=True)

    output_dir = str(output_dir.resolve())

    groups = groups - 1  # adjust for matlab starting index = 1

    if make_video_parallel_or_series == 'parallel':
        for i in range(len(groups)):
            make_brady_movie_parallel(i, destination_path=output_dir, groups=groups, clips=clips)
    else:
        for i in range(len(groups)):
            make_brady_movie_series(i, destination_path=output_dir, groups=groups, clips=clips)
