"""
Adapted the code from DeepLabCut2.2 Toolbox (deeplabcut.org)
https://github.com/AlexEMG/DeepLabCut


"""

import os
import shutil
from pathlib import Path
import yaml
from autoposemapper.setRunParameters import set_run_parameter


def create_new_project(project,
                       experimenter,
                       videos,
                       working_directory=None,
                       copy_videos=False,
                       video_type='.mp4',
                       sleap_or_dlc_or_conv='sleap'):
    """Creates a new project directory, sub-directories and a basic configuration file.
    The configuration file is loaded with the default values. Change its parameters to your projects need.

    Parameters
    ----------
    project : string
        the name of the project.
    experimenter : string
        the name of the experimenter.
    videos : list
        A list of string containing the full paths of the videos to include in the project.
        Attention: Can also be a directory, then all videos of video_type will be imported.
    working_directory : string, optional
        The directory where the project will be created. The default is the ``current working directory``;
        if provided, it must be a string.
    copy_videos: Boolean:
        Whether the copy the video files or not
    video_type: basestring
        The video type can be an ".mp4" or ".avi"
    sleap_or_dlc_or_conv: string
        Will use sleap or dlc for tracking

    Example
    --------
    Linux/MacOs
    # >>>autoposemapper.create_new_project('cohabitation','John',['/data/videos/vol1.avi','/data/videos/vole2.avi'],
    '/analysis/project/')
    # >>> autoposemapper.create_new_project('cohabitation','John',['/data/videos'],video_type='.mp4')
    Windows:
    # >>> autoposemapper.create_new_project('cohabitation','Jane',[r'C:\yourusername\Videos\vole1.avi'],
    copy_videos=True)
    Users must format paths with either:  r'C:\ OR 'C:\\ <- i.e. a double backslash \ \ )
    """

    from datetime import datetime as dt
    parameters = set_run_parameter()

    months_3letter = {
        1: "Jan",
        2: "Feb",
        3: "Mar",
        4: "Apr",
        5: "May",
        6: "Jun",
        7: "Jul",
        8: "Aug",
        9: "Sep",
        10: "Oct",
        11: "Nov",
        12: "Dec",
    }

    date = dt.today()
    month = months_3letter[date.month]
    day = date.day
    d = str(month[0:3] + str(day))
    date = dt.today().strftime("%Y-%m-%d")
    if working_directory is None:
        working_directory = "."
    wd = Path(working_directory).resolve()
    project_name = "{pn}-{exp}-{date}".format(pn=project, exp=experimenter, date=date)
    project_path = wd / project_name

    # Create project and subdirectories
    if project_path.exists():
        print(f'Project {project_path} already exists')
        return

    config_path = project_path / parameters.config_name

    video_path = project_path / parameters.video_path_name
    sleap_data = project_path / parameters.sleap_data_name
    dlc_data = project_path / parameters.dlc_data_name
    autoencoder_data = project_path / parameters.autoencoder_data_name
    id_tracker_data = project_path / parameters.id_tracker_data_name
    bash_files = project_path / parameters.bash_files_name
    conv_autoencoder_data = project_path / parameters.conv_autoencoder_data_name

    if sleap_or_dlc_or_conv == 'sleap':
        for p in [video_path, sleap_data, autoencoder_data, bash_files]:
            p.mkdir(parents=True, exist_ok=True)
            print(f'Created {p}')
    elif sleap_or_dlc_or_conv == 'dlc':
        for p in [video_path, dlc_data, autoencoder_data, bash_files]:
            p.mkdir(parents=True, exist_ok=True)
            print(f'Created {p}')
    elif sleap_or_dlc_or_conv == 'conv':
        for p in [video_path, conv_autoencoder_data, autoencoder_data, bash_files]:
            p.mkdir(parents=True, exist_ok=True)
            print(f'Created {p}')
    else:
        for p in [video_path, sleap_data, dlc_data, autoencoder_data, id_tracker_data, bash_files,
                  conv_autoencoder_data]:
            p.mkdir(parents=True, exist_ok=True)
            print(f'Created {p}')

    vids = []
    for i in videos:
        # Check if it is a folder
        if os.path.isdir(i):
            vids_in_dir = [
                os.path.join(i, vp) for vp in os.listdir(i) if vp.endswith(video_type)
            ]
            vids = vids + vids_in_dir
            if len(vids_in_dir) == 0:
                print("No videos found in", i)
                print(
                    "Perhaps change the video_type, which is currently set to:",
                    video_type,
                )
            else:
                videos = vids
                print(
                    len(vids_in_dir),
                    " videos from the directory",
                    i,
                    "were added to the project.",
                )
        else:
            if os.path.isfile(i):
                vids = vids + [i]
            videos = vids

    videos = [Path(vp) for vp in videos]

    destinations = [video_path.joinpath(f"{vp.stem}/{vp.name}") for vp in videos]
    for folder in destinations:
        folder_parent = folder.parents[0]
        if not folder_parent.exists():
            folder_parent.mkdir(parents=True)

    if copy_videos:
        print("Copying the videos")
        for src, dst in zip(videos, destinations):
            shutil.copy(
                os.fspath(src), os.fspath(dst)
            )  # https://www.python.org/dev/peps/pep-0519/
    else:
        # creates the symlinks of the video and puts it in the videos' directory.
        print("Attempting to create a symbolic link of the video ...")
        for src, dst in zip(videos, destinations):
            if dst.exists():
                raise FileExistsError("Video {} exists already!".format(dst))
            try:
                src = str(src)
                dst = str(dst)
                os.symlink(src, dst)
                print("Created the symlink of {} to {}".format(src, dst))
            except OSError:
                try:
                    import subprocess

                    subprocess.check_call("mklink %s %s" % (dst, src), shell=True)
                except OSError:
                    print(
                        "Symlink creation impossible (exFat architecture?): "
                        "cutting/pasting the video instead."
                    )
                    shutil.move(os.fspath(src), os.fspath(dst))
                    print("{} moved to {}".format(src, dst))
            videos = destinations

    if sleap_or_dlc_or_conv == 'sleap':
        destinations_cnn = [sleap_data.joinpath(f"{vp.stem}/{vp.name}") for vp in videos]
        for folder in destinations_cnn:
            folder_parent = folder.parents[0]
            if not folder_parent.exists():
                folder_parent.mkdir(parents=True)
        if copy_videos:
            print("Copying the videos")
            for src, dst in zip(videos, destinations_cnn):
                shutil.copy(
                    os.fspath(src), os.fspath(dst)
                )
        else:
            # creates the symlinks of the video and puts it in the videos' directory.
            print("Attempting to create a symbolic link of the video ...")
            for src, dst in zip(videos, destinations_cnn):
                if dst.exists():
                    raise FileExistsError("Video {} exists already!".format(dst))
                try:
                    src = str(src)
                    dst = str(dst)
                    os.symlink(src, dst)
                    print("Created the symlink of {} to {}".format(src, dst))
                except OSError:
                    try:
                        import subprocess

                        subprocess.check_call("mklink %s %s" % (dst, src), shell=True)
                    except OSError:
                        print(
                            "Symlink creation impossible (exFat architecture?): "
                            "cutting/pasting the video instead."
                        )
                        shutil.move(os.fspath(src), os.fspath(dst))
                        print("{} moved to {}".format(src, dst))

    elif sleap_or_dlc_or_conv == 'dlc':
        destinations_cnn = [dlc_data.joinpath(f"{vp.stem}/{vp.name}") for vp in videos]
        for folder in destinations_cnn:
            folder_parent = folder.parents[0]
            if not folder_parent.exists():
                folder_parent.mkdir(parents=True)
        if copy_videos:
            print("Copying the videos")
            for src, dst in zip(videos, destinations_cnn):
                shutil.copy(
                    os.fspath(src), os.fspath(dst)
                )
        else:
            # creates the symlinks of the video and puts it in the videos' directory.
            print("Attempting to create a symbolic link of the video ...")
            for src, dst in zip(videos, destinations_cnn):
                if dst.exists():
                    raise FileExistsError("Video {} exists already!".format(dst))
                try:
                    src = str(src)
                    dst = str(dst)
                    os.symlink(src, dst)
                    print("Created the symlink of {} to {}".format(src, dst))
                except OSError:
                    try:
                        import subprocess

                        subprocess.check_call("mklink %s %s" % (dst, src), shell=True)
                    except OSError:
                        print(
                            "Symlink creation impossible (exFat architecture?): "
                            "cutting/pasting the video instead."
                        )
                        shutil.move(os.fspath(src), os.fspath(dst))
                        print("{} moved to {}".format(src, dst))
    elif sleap_or_dlc_or_conv == 'conv':
        destinations_cnn = [conv_autoencoder_data.joinpath(f"{vp.stem}/{vp.name}") for vp in videos]
        for folder in destinations_cnn:
            folder_parent = folder.parents[0]
            if not folder_parent.exists():
                folder_parent.mkdir(parents=True)
        if copy_videos:
            print("Copying the videos")
            for src, dst in zip(videos, destinations_cnn):
                shutil.copy(
                    os.fspath(src), os.fspath(dst)
                )
        else:
            # creates the symlinks of the video and puts it in the videos' directory.
            print("Attempting to create a symbolic link of the video ...")
            for src, dst in zip(videos, destinations_cnn):
                if dst.exists():
                    raise FileExistsError("Video {} exists already!".format(dst))
                try:
                    src = str(src)
                    dst = str(dst)
                    os.symlink(src, dst)
                    print("Created the symlink of {} to {}".format(src, dst))
                except OSError:
                    try:
                        import subprocess

                        subprocess.check_call("mklink %s %s" % (dst, src), shell=True)
                    except OSError:
                        print(
                            "Symlink creation impossible (exFat architecture?): "
                            "cutting/pasting the video instead."
                        )
                        shutil.move(os.fspath(src), os.fspath(dst))
                        print("{} moved to {}".format(src, dst))

    if copy_videos:
        videos = destinations  # in this case the *new* location should be added to the config file

    # adds the video list to the config.yaml file
    video_sets = []
    for video in videos:
        print(video)
        try:
            # For Windows os.path.realpath does not work and does not link to the real video. [old: rel_video_path =
            # os.path.realpath(video)]
            rel_video_path = str(Path.resolve(Path(video)))
        except OSError:
            rel_video_path = os.readlink(str(video))
        video_sets.append(rel_video_path)

    cfg_file = {"Task": project, "scorer": experimenter, "video_sets": video_sets, "project_path": str(project_path),
                "date": d}
    # common parameters:

    with open(config_path, "w") as outfile:
        yaml.dump(cfg_file, outfile)

    print('Generated "{}"'.format(project_path / "config.yaml"))
    return project_path.resolve()
