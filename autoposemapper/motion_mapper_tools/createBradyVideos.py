import glob
import os
from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
import hdf5storage
from pathlib import Path
import re
import yaml
import pandas as pd


def create_brady_videos(project_path, watershed_path, autoencoder_data_path, video_path, encoder_type='SAE'):
    gp_data = str(Path(watershed_path).resolve())
    groups_data = hdf5storage.loadmat(gp_data)
    groups = abs(groups_data['groups'])
    zValNames = groups_data['zValNames'][0]

    auto_files = sorted(glob.glob(f"{autoencoder_data_path}/**/*{encoder_type}*.h5", recursive=True))
    h5sfnames = []
    counter = 0

    # Creating full path for h5 files created by Autoencoder
    for val in zValNames:
        file_name = val[0][0]
        file_name1 = file_name[:file_name.find('_animal')]
        file_name2 = file_name[file_name.find('animal_'):]
        file_name2 = file_name2.rsplit('_', 2)[0]
        for file in auto_files:
            if re.search(file_name1, file) and re.search(file_name2, file):
                h5sfnames.append(file)
                counter += 1
                break

    # Check to make sure all files exist
    count = 0
    for i in h5sfnames:
        if os.path.exists(i):
            count += 1
    assert counter == count

    # Creating full path for videos files¶
    auto_files = sorted(glob.glob(f"{autoencoder_data_path}/**/*{encoder_type}*.h5", recursive=True))
    vidnames = []
    counter = 0

    # Creating full path for h5 files created by Autoencoder
    for val in zValNames:
        file_name = val[0][0]
        file_name1 = file_name[:file_name.find('_animal')]
        file_name2 = file_name[file_name.find('animal_'):]
        file_name2 = file_name2.rsplit('_', 2)[0]
        for file in auto_files:
            if re.search(file_name1, file) and re.search(file_name2, file):
                vid_name = glob.glob(f'{video_path}**/{file_name1}.mp4', recursive=True)[0]
                vidnames.append(vid_name)
                counter += 1
                break

    # Check to make sure all files exist
    count = 0
    for i in vidnames:
        if os.path.exists(i):
            count += 1
        else:
            print(i)
    assert counter == count

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

    # Connections point on the rodent
    skeleton_path = Path(project_path) / 'skeleton_ego.yaml'
    with open(skeleton_path, 'r') as f:
        sk = yaml.load(f, Loader=yaml.FullLoader)
    connect = sk['Skeleton']
    connections = list(connect)

    new_h5s = [pd.read_hdf(h5f).values for h5f in h5sfnames]

    new_h5s = [h5.reshape((h5.shape[0], int(h5.shape[1] / 2), 2)) for h5 in new_h5s]

    return output_dir, groups, connections, new_h5s, vidnames


def center_video(region, output_dir, groups, connections, h5s, vidnames, animal_fps=25, subs=4, num_pad=250):

    output_name = output_dir + '/regions_' + '%.3i' % (region + 1) + '.mp4'
    if os.path.exists(output_name):
        return

    if not groups[region][0].shape[0]:
        return
    # number of plots to make
    nplots = min(subs * subs, groups[region][0].shape[0])

    # randomly select videos to make brady movies from the videos in the group list
    selectedclips = np.random.choice([i for i in range(groups[region][0].shape[0])], size=nplots, replace=False)
    vidindslist = groups[region][0][selectedclips, 0]

    framestoplot = [np.arange(groups[region][0][i, 1], groups[region][0][i, 2])
                    for i in selectedclips]

    # the max number of frames from the selected videos
    maxsize = min([i.shape[0] for i in framestoplot])

    # Make the range of frames from the selected videos the same length
    framestoplot = np.array([np.resize(i, maxsize) for i in framestoplot])

    subx = max(2, int(np.ceil(np.sqrt(nplots))))

    fig, axes = plt.subplots(subx, subx, figsize=(16, 16))
    fig.subplots_adjust(0, 0, 1.0, 1.0, 0.0, 0.0)
    lines = []
    ims = []
    print(vidindslist)

    for i in range(subx * subx):
        ax = axes[i // subx, i % subx]
        ax.axis('off')

        if i >= nplots:
            continue
        # Create range of x and y values for the frames
        xmin = np.round(h5s[vidindslist[i]][framestoplot[i, 0], 4, 0]).astype('int') - num_pad
        xmax = np.round(h5s[vidindslist[i]][framestoplot[i, 0], 4, 0]).astype('int') + num_pad
        ymin = np.round(h5s[vidindslist[i]][framestoplot[i, 0], 4, 1]).astype('int') - num_pad
        ymax = np.round(h5s[vidindslist[i]][framestoplot[i, 0], 4, 1]).astype('int') + num_pad
        for conn in connections:
            lines.append(ax.plot(h5s[vidindslist[i]][framestoplot[i, 0], conn, 0] - xmin,
                                 h5s[vidindslist[i]][framestoplot[i, 0], conn, 1] - ymin)[0])
        clip_path = vidnames[vidindslist[i]]
        clip = VideoFileClip(clip_path)
        frame = clip.get_frame((framestoplot[i, 0]) / 30.0)

        # set_trace()
        # Padding the frames
        frame_region = np.pad(frame, ((num_pad, num_pad), (num_pad, num_pad), (0, 0)), 'minimum')
        frame_region = frame_region[num_pad + ymin:num_pad + ymax, num_pad + xmin:num_pad + xmax, :]
        if not frame_region.any():
            return
        ims.append(ax.imshow(frame_region))
        #         ax.text(0,40,str(vidindslist[i]),color='red', fontsize=20)
        ax.set_aspect(1.0)

    def animate(t):
        for lnum, line in enumerate(lines):
            i = lnum // len(connections)
            j = lnum % len(connections)
            # Change the 8 to the actual body center of the animal
            if not j:
                xmin = np.round(h5s[vidindslist[i]][framestoplot[i, 0], 4, 0]).astype('int') - num_pad
                xmax = np.round(h5s[vidindslist[i]][framestoplot[i, 0], 4, 0]).astype('int') + num_pad
                ymin = np.round(h5s[vidindslist[i]][framestoplot[i, 0], 4, 1]).astype('int') - num_pad
                ymax = np.round(h5s[vidindslist[i]][framestoplot[i, 0], 4, 1]).astype('int') + num_pad
                clip_path = vidnames[vidindslist[i]]
                clip = VideoFileClip(clip_path)
                frame = clip.get_frame((framestoplot[i, t]) / 30.0)
                frame_region = np.pad(frame, ((num_pad, num_pad), (num_pad, num_pad), (0, 0)), 'minimum')
                frame_region = frame_region[num_pad + ymin:num_pad + ymax, num_pad + xmin:num_pad + xmax, :]
                ims[i].set_array(frame_region)
            line.set_data(h5s[vidindslist[i]][framestoplot[i, t], connections[j], 0] - xmin,
                          h5s[vidindslist[i]][framestoplot[i, t], connections[j], 1] - ymin)

        return tuple(lines + ims)

    line_ani = animation.FuncAnimation(fig, animate, maxsize, repeat=True)

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=animal_fps, metadata=dict(artist='Sena'), bitrate=1800)
    line_ani.save(output_dir + '/regions_' + '%.3i' % (region + 1) + '.mp4', writer=writer)
    plt.close()
    print(region + 1, ' done')
