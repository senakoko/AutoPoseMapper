import glob
import os
import pickle
from datetime import datetime
from scipy.io import loadmat
import hdf5storage
from pathlib import Path

from autoposemapper import motionmapperpy as mmpy


def run_motion_mapper(encoded_path,
                      minF=0.5,
                      maxF=15,
                      perplexity=32,
                      tSNE_method='barnes_hut',
                      samplingFreq=30,
                      numPeriods=50,
                      numProcessors=4,
                      useGPU=-1,
                      training_numPoints=5000,
                      trainingSetSize=50000,
                      embedding_batchSize=30000
                      ):

    tsne_file_name = (Path(encoded_path).parents[0] / 'TSNE').resolve()
    if not tsne_file_name.exists():
        tsne_file_name.mkdir(parents=True)

    parameters = mmpy.setRunParameters()

    # change this taskFolder path to where the projections are. The path should be the parent path to
    # the projection folder. Don't include the '/' at the end

    taskFolder = Path(encoded_path).parents[0]

    parameters.minF = minF  # % Minimum frequency for Morlet Wavelet Transform

    parameters.maxF = maxF  # % Maximum frequency for Morlet Wavelet Transform,
    # % equal to Nyquist frequency for your measurements.

    parameters.perplexity = perplexity  # %2^H (H is the transition entropy)

    parameters.tSNE_method = tSNE_method  # Global tSNE method - 'barnes_hut' or 'exact'

    parameters.samplingFreq = samplingFreq  # % Sampling frequency (or FPS) of data.

    parameters.numPeriods = numPeriods  # % No. of frequencies between minF and maxF.

    parameters.numProcessors = numProcessors  # % No. of processor to use when parallel
    # % processing (for wavelets, if not using GPU). -1 to use all cores.

    parameters.useGPU = useGPU  # GPU to use, set to -1 if GPU not present

    parameters.training_numPoints = training_numPoints  # % Number of points in mini-tSNEs.

    # %%%%% NO NEED TO CHANGE THESE UNLESS RAM (NOT GPU) MEMORY ERRORS RAISED%%%%%%%%%%
    parameters.trainingSetSize = trainingSetSize  # % Total number of representative points to find.
    # Increase or decrease based on available RAM. For reference, 36k is a good number with 64GB RAM.

    parameters.embedding_batchSize = embedding_batchSize  # % Lower this if you get a memory error
    # when re-embedding points on learned tSNE map.

    taskFolder = str(taskFolder)

    projectionFiles = glob.glob(taskFolder + '/Projections/*pcaModes.mat')

    m = loadmat(projectionFiles[0], variable_names=['projections'])['projections']

    parameters.pcaModes = m.shape[1]  # %Number of PCA projections in saved files.
    print(parameters.pcaModes)
    parameters.numProjections = parameters.pcaModes

    print(datetime.now().strftime('%m-%d-%Y_%H-%M'))
    print('tsneStarted')

    if not os.path.exists(taskFolder + '/TSNE/training_tsne_embedding.mat'):
        print('Running minitSNE')
        mmpy.subsampled_tsne_from_projections(parameters, taskFolder)
        print('minitSNE done, finding embeddings now.')
        print(datetime.now().strftime('%m-%d-%Y_%H-%M'))

    import h5py
    with h5py.File(taskFolder + '/TSNE/training_data.mat', 'r') as hfile:
        trainingSetData = hfile['trainingSetData'][:].T
    with h5py.File(taskFolder + '/TSNE/training_tsne_embedding.mat', 'r') as hfile:
        trainingEmbedding = hfile['trainingEmbedding'][:].T

    for i in range(len(projectionFiles)):
        print('Finding Embeddings')
        print('%i/%i : %s' % (i + 1, len(projectionFiles), projectionFiles[i]))
        if os.path.exists(projectionFiles[i][:-4] + '_zVals.mat'):
            print('Already done. Skipping.\n')
            continue

        projections = loadmat(projectionFiles[i])['projections']
        zValues, outputStatistics = mmpy.findEmbeddings(projections, trainingSetData, trainingEmbedding, parameters)

        hdf5storage.write(data={'zValues': zValues}, path='/', truncate_existing=True,
                          filename=projectionFiles[i][:-4] + '_zVals.mat', store_python_metadata=False,
                          matlab_compatible=True)
        with open(projectionFiles[i][:-4] + '_zVals_outputStatistics.pkl', 'wb') as hfile:
            pickle.dump(outputStatistics, hfile)

        print('Embeddings saved.\n')
        del zValues, projections, outputStatistics

    print('All Embeddings Saved!')

    mmpy.findWatershedRegions(taskFolder, parameters, minimum_regions=100, startsigma=0.3, pThreshold=[0.33, 0.67],
                              saveplot=True, endident='*_pcaModes.mat')
