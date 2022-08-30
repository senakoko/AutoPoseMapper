import glob
import pandas as pd
import numpy as np
from scipy.io import savemat, loadmat
from pathlib import Path
from autoposemapper.setRunParameters import set_run_parameter


def extract_projections(encoded_path):
    parameters = set_run_parameter()

    h5_path = Path(encoded_path)
    h5_files = sorted(glob.glob(f"{h5_path}/**/*.h5", recursive=True))

    projection_path = Path(encoded_path).parents[0] / parameters.projection_name
    if not projection_path.exists():
        projection_path.mkdir(parents=True)

    projection_path = projection_path.resolve()

    for file in h5_files:
        print(Path(file).stem)
        base_c_df = pd.read_hdf(file)
        base_c_df.replace(to_replace=[0], method='ffill', inplace=True)
        base_c_df.replace(to_replace=[0], method='bfill', inplace=True)
        base_c = base_c_df.values
        savemat(f"{projection_path}/{Path(file).stem}_pcaModes.mat", {'projections': base_c})
        del base_c, base_c_df


def check_extracted_projections(encoded_path):
    parameters = set_run_parameter()

    projection_path = Path(encoded_path).parents[0] / parameters.projection_name
    projection_files = sorted(glob.glob(f"{projection_path}/*pcaModes.mat", recursive=True))

    data_count = []
    counter = 0
    for i in projection_files:
        # print(i)
        m = loadmat(i, variable_names=['projections'])['projections']
        # print(m.shape)
        data_count.append(m.shape[0])
        counter += 1
    print('the number of files found: ', counter)
    print('the total number of data points to be used in the behavioral space: ', np.sum(data_count))
