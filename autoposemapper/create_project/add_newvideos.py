import glob
from pathlib import Path
import os
import shutil
import yaml
from autoposemapper.setRunParameters import set_run_parameter


def add_new_videos(project_path, path2video, video_type='.mp4',
                   copy_videos=False, sleap_or_dlc_or_conv='sleap'):
    """
    Add new videos to the main project folder
    :param project_path: (string) the path to the project folder
    :param path2video: (string) the path to the video files
    :param video_type: (string) the name extension for the video files
    :param copy_videos: (bool) whether to copy the videos or not
    :param sleap_or_dlc_or_conv: (string) Used sleap or dlc for tracking
    :return:
    """
    parameters = set_run_parameter()

    project_path = Path(project_path)
    config_path = project_path / parameters.config_name
    video_path = project_path / parameters.video_path_name
    sleap_data = project_path / parameters.sleap_data_name
    dlc_data = project_path / parameters.dlc_data_name
    conv_autoencoder_data = project_path / parameters.conv_autoencoder_data_name

    if os.path.isdir(path2video):
        path2video = glob.glob(f"{path2video}/**/*{video_type}", recursive=True)
        videos = [Path(vp) for vp in path2video]
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
                    print("Video {} exists already!".format(dst))
                    continue
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
                        print("Video {} exists already!".format(dst))
                        continue
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
                        print("Video {} exists already!".format(dst))
                        continue
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
                        print("Video {} exists already!".format(dst))
                        continue
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

    elif os.path.isfile(path2video):
        path2video = [path2video]
        videos = [Path(vp) for vp in path2video]
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
                    print("Video {} exists already!".format(dst))
                    continue
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
                        print("Video {} exists already!".format(dst))
                        continue
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
                        print("Video {} exists already!".format(dst))
                        continue
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
                        print("Video {} exists already!".format(dst))
                        continue
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

    with open(config_path, "r") as outfile:
        config = yaml.safe_load(outfile)
        video_sets = config['video_sets']
        config.pop('video_sets')

    # adds the video list to the config.yaml file
    for video in videos:
        print(video)
        try:
            # For Windows os.path.realpath does not work and does not link to the real video. [old: rel_video_path =
            # os.path.realpath(video)]
            rel_video_path = str(Path.resolve(Path(video)))
        except OSError:
            rel_video_path = os.readlink(str(video))
        video_sets.append(rel_video_path)

    config["video_sets"] = video_sets

    with open(config_path, "w+") as outfile:
        yaml.safe_dump(config, outfile, explicit_start=True)
