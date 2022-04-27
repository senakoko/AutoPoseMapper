import numpy as np
import pandas as pd
from scipy.io import savemat
from tqdm import tqdm


def egocenter_h5(h5file, bind_center=None, b1=None, b2=None,
                 silent=False, tracker="CNN", drop_point=True, which_points=['tailEnd']):
    """
    Rotate and egocenter the tracked points
    parameter
    ----------
    h5file: the pandas h5 file.
    bind_center: string -  the middle point to use in ego-centering
    b1: string - the start point to use to ego-center
    b2: string - the end point to use to ego-center
    silent: boolean -  Show the process of ego-centering the data
    tracker: string - the name of the tracker: default CNN to represent both SLEAP and maDLC
    drop_point: boolean - drop point(s) because they might be unstable to track
    which_points: string - a list of points to drop
    return: ego-centered mat file data
    """
    data = pd.read_hdf(h5file)
    data.interpolate(inplace=True)
    data.fillna(method='bfill', inplace=True)

    scorer = data.columns.get_level_values('scorer').unique().item()
    individuals = data.columns.get_level_values('individuals').unique().to_list()
    body_parts = data.columns.get_level_values('bodyparts').unique().to_list()

    if drop_point:
        if len(which_points) == 0:
            point_number = int(input('How many points do you want to drop'))
            which_points = []
            for pn in range(point_number):
                which_p = input('Which points do you want to drop. Enter one at a time and hit enter: ')
                which_points.append(which_p)

        for bpd in which_points:
            body_parts.remove(bpd)

    bpts_val = {}
    for i, bpts in enumerate(body_parts):
        bpts_val[bpts] = i

    if bind_center is None:
        bind_center = input('Enter the name of the middle point to use in ego-centering')
    if b1 is None:
        b1 = input('Enter the name of the start point to use in ego-centering')
    if b2 is None:
        b2 = input('Enter the name of the start point to use in ego-centering')

    bind_center_value = bpts_val[bind_center]
    b1_value = bpts_val[b1]
    b2_value = bpts_val[b2]

    for v, individual in enumerate(individuals):

        individual_data = data[scorer][individual]
        for points in which_points:
            individual_data = individual_data.drop(points, axis=1)

        # Removing the tail points from the dataset
        vals = individual_data.values

        # Transform to 3D
        h5 = vals.reshape((vals.shape[0], int(vals.shape[1] / 2), 2))

        # This is not the best but tqdm always has this error that module is not callable

        # returns all points except bind center
        ginds = np.setdiff1d(np.arange(h5.shape[1]), bind_center_value)
        # subtracts bind center from each point
        ego_h5 = h5[:, :, :2] - h5[:, [bind_center_value for i in range(h5.shape[1])], :2]
        # index all points except for the bind center
        ego_h5 = ego_h5[:, ginds]

        dir_arr = ego_h5[:, b1_value] - ego_h5[:, b2_value - 1]
        # creates the directional array
        dir_arr = dir_arr / np.linalg.norm(dir_arr, axis=1)[:, np.newaxis]

        if not silent:
            for t in tqdm(range(ego_h5.shape[0])):
                # creates a 2x2 of the mid-point between b1 and b2
                rot_mat = np.array([[dir_arr[t, 0], dir_arr[t, 1]], [-dir_arr[t, 1], dir_arr[t, 0]]])
                # dot multiply each body part by the mid-point between b1 and b2 to perform rotation
                ego_h5[t] = np.array(np.dot(ego_h5[t], rot_mat.T))
        elif silent:
            for t in range(ego_h5.shape[0]):
                rot_mat = np.array([[dir_arr[t, 0], dir_arr[t, 1]], [-dir_arr[t, 1], dir_arr[t, 0]]])
                ego_h5[t] = np.array(np.dot(ego_h5[t], rot_mat.T))

        h5rs_2D = ego_h5.reshape(ego_h5.shape[0], (ego_h5.shape[1] * ego_h5.shape[2]))

        a = h5file[:h5file.find(f'_{tracker}')]
        savemat(f"{a}_{tracker}_animal_{v + 1}_data.mat", {'animal_d': vals})
        savemat(f"{a}_{tracker}_ego_animal_{v + 1}_data.mat", {'animal_d': h5rs_2D})

    body_parts.remove(bind_center)
    bpts_val = {}
    for i, bpts in enumerate(body_parts):
        bpts_val[bpts] = i

    return bind_center_value, b1_value, b2_value, body_parts, bpts_val
