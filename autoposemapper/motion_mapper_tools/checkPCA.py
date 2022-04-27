import glob
import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.decomposition import PCA
from pathlib import Path


def check_pca(encoded_path):

    h5_path = Path(encoded_path)
    h5_files = sorted(glob.glob(f"{h5_path}/**/*.h5", recursive=True))

    pca = PCA(n_components=0.95)

    n_pca_components = []
    for file in tqdm(h5_files):
        # print(file)
        h5 = pd.read_hdf(file)
        pca.fit(h5.values)
        n_pca_components.append(pca.n_components_)

    print('the maximum number of principal components in the data: ', np.max(n_pca_components))
    print('the median number of principal components in the data: ', np.median(n_pca_components))
