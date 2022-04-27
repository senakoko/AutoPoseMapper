import os
import sys
import numpy as np
from pathlib import Path
from skimage import io
from skimage.util import img_as_ubyte
import cv2
from tqdm import tqdm
from autoposemapper.convolutional_autoencoder import frameselectiontools


def extract_frames(video, output_path=None, numframes2pick=100, algo="uniform", userfeedback=True, cluster_step=1,
                   cluster_resizewidth=30, cluster_color=False, opencv=True, name_prefix='extracted'):
    """
    Extracts frames from the videos in the file path.

    The provided function either selects frames from the videos in a randomly and temporally uniformly distributed way
    (uniform), \n by clustering based on visual appearance (k-means), or by manual selection.

    Parameters
    ----------
    video: Video to extract frames from.
    output_path: destination path for extracted frames
    numframes2pick: the number of frames to extract from the video
    algo : string
        Specifying the algorithm to use for selecting the frames. Currently, it supports either ``kmeans`` or
        ``uniform`` based selection. This flag is
        only required for ``automatic`` mode and the default is ``uniform``. For uniform, frames are picked in
        temporally uniform way, kmeans performs clustering on down-sampled frames (see user guide for details).
        Note: color information is discarded for kmeans, thus e.g. for camouflaged octopus clustering one
        might want to change this.
    userfeedback: bool, optional
        If this is set to false during automatic mode then frames for all videos are extracted. The user can set this
        to true, which will result in a dialog, where the user is asked for each video if (additional/any) frames
        from this video should be extracted. Use this, e.g. if you have already labeled
        some folders and want to extract data for new videos.cluster_resizewidth: number, default: 30
        For k-means one can change the width to which the images are down-sampled (aspect ratio is fixed).
    cluster_step: number, default: 1
        By default each frame is used for clustering, but for long videos one could only use every nth frame
        (set by: cluster_step). This saves memory before clustering can start, however,
        reading the individual frames takes longer due to the skipping.
    cluster_resizewidth: for the kmeans cluster
    cluster_color: bool, default: False
        If false then each down-sampled image is treated as a grayscale vector (discarding color information).
        If true, then the color channels are considered.
        This increases the computational complexity.
    opencv: bool, default: True
        Uses openCV for loading & extracting (otherwise moviepy (legacy))
    name_prefix: prefix appended to the end of the filename.

    Adapted from DeepLabCut2.0 Toolbox (deeplabcut.org), © A. & M. Mathis Labs
    """

    numframes2pick = numframes2pick
    start = 0
    stop = 1

    # Check for variable correctness
    if start > 1 or stop > 1 or start < 0 or stop < 0 or start >= stop:
        raise Exception("Erroneous start or stop values. Please correct it in the config file.")
    if numframes2pick < 1 and not int(numframes2pick):
        raise Exception("Perhaps consider extracting more, or a natural number of frames.")

    if opencv:
        from cv2 import VideoCapture
    else:
        from moviepy.editor import VideoFileClip

    has_failed = []
    if userfeedback:
        print("Do you want to extract (perhaps additional) frames for video:", video, "?")
        askuser = input("yes/no")
    else:
        askuser = 'yes'

    if askuser == 'y' or askuser == 'yes':
        if opencv:
            cap = VideoCapture(video)
            nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        else:
            # Moviepy
            clip = VideoFileClip(video)
            fps = clip.fps
            nframes = int(np.ceil(clip.duration * 1.0 / fps))
        if not nframes:
            print("Video could not be opened. Skipping...")
            # continue

        if np.log10(nframes) == np.inf:
            indexlength = int(1e6)
        else:
            indexlength = int(np.ceil(np.log10(nframes)))

        fname = Path(video)
        if output_path is None:
            output_path = Path(video).parents[0] / f"{name_prefix}-data" / fname.stem
        else:
            output_path = Path(output_path) / f"{name_prefix}-data" / fname.stem

        if not output_path.exists():
            output_path.mkdir(parents=True)

        if output_path.exists():
            if len(os.listdir(output_path)):
                if userfeedback:
                    askuser = input("The directory already contains some frames. Do you want to add to it?(yes/no): ")
                if not (askuser == 'y' or askuser == 'yes' or askuser == "Y" or askuser == "Yes"):
                    sys.exit("Delete the frames and try again later!")

        print("Extracting frames based on %s ..." % algo)
        if algo == 'uniform':
            if opencv:
                frames2pick = frameselectiontools.UniformFramescv2(cap, numframes2pick, start, stop)
            else:
                frames2pick = frameselectiontools.UniformFrames(clip, numframes2pick, start, stop)
        elif algo == 'kmeans':
            if opencv:
                frames2pick = frameselectiontools.KmeansbasedFrameselectioncv2(cap, numframes2pick, start, stop,
                                                                               step=cluster_step,
                                                                               resizewidth=cluster_resizewidth,
                                                                               color=cluster_color)
            else:
                frames2pick = frameselectiontools.KmeansbasedFrameselection(clip, numframes2pick, start, stop,
                                                                            step=cluster_step,
                                                                            resizewidth=cluster_resizewidth,
                                                                            color=cluster_color)
        else:
            print(
                "Please implement this method yourself and send us a pull request! Otherwise, "
                "choose 'uniform' or 'kmeans'.")
            frames2pick = []

        if not len(frames2pick):
            print("Frame selection failed...")
            return

        is_valid = []
        if opencv:
            for index in tqdm(frames2pick):
                cap.set(cv2.CAP_PROP_POS_FRAMES, index - 1)  # extract a particular frame
                ret, frame = cap.read()
                if frame is not None:
                    image = img_as_ubyte(frame)
                    image = image[..., ::-1]
                    img_name = (str(output_path) + "/img" + str(index).zfill(indexlength) + ".png")
                    io.imsave(img_name, image)
                    is_valid.append(True)
                else:
                    print("Frame", index, " not found!")
                    is_valid.append(False)
            cap.release()
        else:
            for index in tqdm(frames2pick):
                try:
                    image = img_as_ubyte(clip.get_frame(index * 1.0 / clip.fps))
                    img_name = (str(output_path) + "/img" + str(index).zfill(indexlength) + ".png")
                    io.imsave(img_name, image)
                    is_valid.append(True)
                except FileNotFoundError:
                    print("Frame # ", index, " does not exist.")
                    is_valid.append(False)
            clip.close()
            del clip

        if not any(is_valid):
            has_failed.append(True)
        else:
            has_failed.append(False)

    else:  # NO!
        has_failed.append(False)

    if all(has_failed):
        print("Frame extraction failed. Video files must be corrupted.")
        return
    elif any(has_failed):
        print("Although most frames were extracted, some were invalid.")
    else:
        print("Frames were successfully extracted, for the videos of interest.")
