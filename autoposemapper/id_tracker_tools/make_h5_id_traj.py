import numpy as np
import pandas as pd
import os
from pathlib import Path


def make_id_trajectories(file, destination_path=None):

    # Destination filename
    destination_name = Path(file).parts[-2]
    if destination_path is None:
        destination_path = str(Path(file).parents[0].resolve())
        destination_file = f"{destination_path}/{destination_name}_idtraj.h5"
    else:
        destination_file = f"{destination_path}/{destination_name}_idtraj.h5"

    if os.path.exists(destination_file):
        return

    print(Path(destination_file).name)

    # Load the trajectories.npy file
    traj = np.load(file, allow_pickle=True).item()
    trajectories = traj['trajectories']

    # Convert it to dataframe
    trajectory = trajectories.reshape((trajectories.shape[0], trajectories.shape[1] * trajectories.shape[2]))
    scorer = 'ID_Tracker'
    individuals = []
    for i in range(trajectories.shape[1]):
        individuals.append(f'ind{i + 1}')
    bodyparts = ['Center']
    coords = ['x', 'y']

    col = pd.MultiIndex.from_product([[scorer], individuals, bodyparts, coords],
                                     names=['scorer', 'individuals', 'bodyparts', 'coords'])
    ind = np.arange(trajectory.shape[0])
    animal = pd.DataFrame(trajectory, index=ind, columns=col)

    # Save file
    animal.to_hdf(destination_file, 'animal_d')
