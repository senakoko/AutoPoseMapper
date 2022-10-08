import numpy as np
import pandas as pd
import re
import os
from pathlib import Path
from tqdm import tqdm
from autoposemapper.setRunParameters import set_run_parameter


def cal_animal_area(h5=None, nsample=50, nframes=30, scorer=None):
    """
    Calculate the area of the animal assuming it has an ellipsoid shape

    Parameters
    ----------
    h5: h5 data
    nsample: number of samples to use to calculate the area
    nframes: number of frames to use to calculate the area
    scorer: annotator of the h5 file e.g., CNN, SAE, VAE, CSI
    """
    area_ns = np.random.choice(h5.shape[0], nsample, replace=False)
    area_list = []

    for i in area_ns:
        if i + nframes > h5.shape[0]:
            continue
        nose = h5[scorer].loc[i:i + nframes, 'Nose']
        left_mid = h5[scorer].loc[i:i + nframes, 'leftMidWaist']
        center = h5[scorer].loc[i:i + nframes, 'midBody']
        a_len = nose.sub(center)
        b_len = center.sub(left_mid)
        a_dist = np.linalg.norm(a_len, axis=1)
        b_dist = np.linalg.norm(b_len, axis=1)
        area = np.pi * a_dist * b_dist
        med_area = np.nanmedian(area)
        area_list.append(med_area)

    overall_area = np.nanmedian(area_list)
    return overall_area


def cal_animal_area_multi(h5, nsample=50, nframes=30):
    """
    Calculate the area of the animal assuming it has an ellipsoid shape

    Parameters
    ----------
    h5: h5 data
    nsample: number of samples to use to calculate the area
    nframes: number of frames to use to calculate the area
    """
    area_ns = np.random.choice(h5.shape[0], nsample, replace=False)
    area_list = []

    for i in area_ns:
        if i + nframes > h5.shape[0]:
            continue
        nose = h5.loc[i:i + nframes, 'Nose']
        left_mid = h5.loc[i:i + nframes, 'leftMidWaist']
        center = h5[scorer].loc[i:i + nframes, 'midBody']
        a_len = nose.sub(center)
        b_len = center.sub(left_mid)
        a_dist = np.linalg.norm(a_len, axis=1)
        b_dist = np.linalg.norm(b_len, axis=1)
        area = np.pi * a_dist * b_dist
        med_area = np.nanmedian(area)
        area_list.append(med_area)

    overall_area = np.nanmedian(area_list)
    return overall_area


def cal_bodypart_distances(file=None, destination_path=None, encoder_type='CNN'):
    """
    Calculate the euclidean distances between body parts and returns a dataframe of distances 
    for only one animal

    Expects to read h5 files. Doesn't work for csv files

    Parameters
    ----------
    file: the path to file with the tracked points.
    destination_path: the path to save tracked points.
    encoder_type: either SAE, VAE or CSI
    """
    parameters = set_run_parameter()
    # Destination filename
    destination_name = Path(file).stem
    if re.search(f'_{encoder_type}_', destination_name):
        name = destination_name[destination_name.find('animal'):destination_name.find('_data')]
        destination_name = destination_name[:destination_name.find(f'_{encoder_type}_')]
    # file_parts = Path(file).parts[7:9]
    # fp_name = "_".join(file_parts)
    if destination_path is None:
        destination_path = file.rsplit('/', 1)[0]
        destination_file = f"{destination_path}/{destination_name}_{name}_euc.h5"
    else:
        destination_file = f"{destination_path}/{destination_name}_{name}_euc.h5"

    if os.path.exists(destination_file):
        return
    else:
        # Dataframe for distances for individuals    
        final_distances = pd.DataFrame()

    # Read file
    h5 = pd.read_hdf(file)

    # Get scorer 
    scorer = h5.columns.get_level_values('scorer').unique().item()

    # Get list of bodyparts
    bodyparts = h5.columns.get_level_values('bodyparts').unique()

    # Get list of individuals
    # individuals = h5.columns.get_level_values('individuals').unique()

    anim_area = cal_animal_area(h5, nsample=120, nframes=30, scorer=scorer)

    # The body part pairs to use to calculate the distances
    body_pairs_values = []
    for i in range(len(bodyparts)):
        for j in range(len(bodyparts)):
            if i == j:
                continue
            elif i > j:
                continue
            parts_inside = [i, j]
            body_pairs_values.append(parts_inside)

    # Get the body part names and the corresponding pairs
    body_pairs = {}
    for i in range(len(body_pairs_values)):
        pairs = body_pairs_values[i]
        body_pairs[f'dist_{i}'] = bodyparts[pairs]

    # load the bodyparts
    # for j, ind in enumerate(individuals):

    bodypart_dist = pd.DataFrame()

    for _, k in enumerate(body_pairs):
        bp_pairs = body_pairs[k]
        bp_pair_vals = h5[scorer][bp_pairs]

        # Calculate the euclidean distance between the body parts
        diff_bp = bp_pair_vals[bp_pairs[0]].sub(bp_pair_vals[bp_pairs[1]])
        dist_bp = np.linalg.norm(diff_bp, axis=1)
        bodypart_dist[k] = dist_bp

    final_distances = pd.concat([final_distances, bodypart_dist], ignore_index=True)
    final_distances.interpolate(inplace=True)
    final_distances.fillna(method='bfill', inplace=True)
    final_distances = final_distances.divide(np.sqrt(anim_area))  # Normalize  distance but the area of the voles
    print(f'{destination_name}_{name}:', int(np.sqrt(anim_area)))

    final_distances.to_hdf(destination_file, parameters.animal_key)


def cal_bodypart_distances_multi(file=None, destination_path=None, encoder_type='CNN'):
    """
    Calculate the euclidean distances between body parts and returns a dataframe of distances 
    for only one animal

    Expects to read h5 files. Doesn't work for csv files

    Parameters
    ----------
    file: the path to file with the tracked points.
    destination_path: the path to save tracked points.
    encoder_type: either SAE, VAE or CSI
    """
    parameters = set_run_parameter()
    # Destination filename
    destination_name = Path(file).stem
    destination_name = destination_name[:destination_name.find(encoder_type)]

    name = 'animal_1'
    if destination_path is None:
        destination_path = str(Path(file).parent)
        destination_file = f"{destination_path}/{destination_name}{name}_euc.h5"
    else:
        destination_file = f"{destination_path}/{destination_name}{name}_euc.h5"

    if os.path.exists(destination_file):
        return
    else:
        # Dataframe for distances for individuals    
        final_distances = pd.DataFrame()

    # Read file
    h5 = pd.read_hdf(file)

    # Get scorer 
    scorer = h5.columns.get_level_values('scorer').unique().item()

    # Get list of bodyparts
    bodyparts = h5.columns.get_level_values('bodyparts').unique()

    # Get list of individuals
    individuals = h5.columns.get_level_values('individuals').unique()

    for it, ind in enumerate(tqdm(individuals)):

        ind_h5 = h5[scorer][ind]

        anim_area = cal_animal_area_multi(ind_h5, nsample=120, nframes=30)

        # The body part pairs to use to calculate the distances
        body_pairs_values = []
        for i in range(len(bodyparts)):
            for j in range(len(bodyparts)):
                if i == j:
                    continue
                elif i > j:
                    continue
                parts_inside = [i, j]
                body_pairs_values.append(parts_inside)

        # Get the body part names and the corresponding pairs
        body_pairs = {}
        for i in range(len(body_pairs_values)):
            pairs = body_pairs_values[i]
            body_pairs[f'dist_{i}'] = bodyparts[pairs]

        # load the bodyparts
        # for j, ind in enumerate(individuals):

        bodypart_dist = pd.DataFrame()

        for _, k in enumerate(body_pairs):
            bp_pairs = body_pairs[k]
            bp_pair_vals = h5[scorer][ind][bp_pairs]

            # Calculate the euclidean distance between the body parts
            diff_bp = bp_pair_vals[bp_pairs[0]].sub(bp_pair_vals[bp_pairs[1]])
            dist_bp = np.linalg.norm(diff_bp, axis=1)
            bodypart_dist[k] = dist_bp

        final_distances = pd.concat([final_distances, bodypart_dist], ignore_index=True)
        final_distances.interpolate(inplace=True)
        final_distances.fillna(method='bfill', inplace=True)
        final_distances = final_distances.divide(np.sqrt(anim_area))  # Normalize  distance but the area of the voles

        name = f'animal_{it + 1}'
        if destination_path is None:
            destination_path = file.rsplit('/', 1)[0]
            destination_file = f"{destination_path}/{destination_name}{name}_euc.h5"
        else:
            destination_file = f"{destination_path}/{destination_name}{name}_euc.h5"

        print(f'{destination_name}{name}:', int(np.sqrt(anim_area)))

        final_distances.to_hdf(destination_file, parameters.animal_key)
