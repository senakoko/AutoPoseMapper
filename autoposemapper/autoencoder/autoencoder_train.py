import os
import glob
from pathlib import Path
from scipy.io import savemat, loadmat
from tensorflow import keras as k
import pandas as pd
from autoposemapper.autoencoder.models.stacked_autoencoder import StackedAE
from autoposemapper.autoencoder.models.variational_autoencoder import VariationalAE


class AutoTrain:
    def __init__(self, project_path):
        self.project_path = project_path

    def auto_train_initial(self, scorer_type='CNN', encoder_type='SAE', coding_size=16, epochs=5,
                           batch_size=256, earlystop=10, verbose=1, gpu='0',
                           scaling_factor=10):

        mat_path = Path(self.project_path) / 'autoencoder_data'
        mat_files = sorted(glob.glob(f'{str(mat_path)}/**/*{scorer_type}_ego*.mat', recursive=True))

        data = loadmat(mat_files[0])
        data = data['animal_d']

        checkpoint_path = Path(self.project_path) / 'Training'
        if encoder_type == 'SAE':
            auto = StackedAE(num_feat=data.shape[1], gpu=gpu,
                             epochs=epochs, batch_size=batch_size, coding_size=coding_size,
                             checkpoint_path=checkpoint_path, training_num=f'{encoder_type}_{coding_size}',
                             earlystop=earlystop, verbose=verbose)

        elif encoder_type == 'VAE':
            auto = VariationalAE(num_feat=data.shape[1], gpu=gpu,
                                 epochs=epochs, batch_size=batch_size, coding_size=coding_size,
                                 checkpoint_path=checkpoint_path, training_num=f'{encoder_type}_{coding_size}',
                                 earlystop=earlystop, scaling_factor=scaling_factor, verbose=verbose)

        for mat_file in mat_files:
            mat_p = Path(mat_file).resolve()
            print(mat_p.stem)
            data = loadmat(mat_file)
            data = data['animal_d']
            train_model = auto.train(data)

        training_model_path = (Path(self.project_path) / 'Training' / 'models' /
                               f'model_{encoder_type}_{coding_size}_0')

        if training_model_path.exists():
            num = int(str(training_model_path).rsplit("_")[-1])
            training_model_path = (Path(self.project_path) / 'Training' /
                                   'models' / f'model_{encoder_type}_{coding_size}_{num + 1}')
        auto.model.save(training_model_path)

        training_weight_path = (Path(self.project_path) / 'Training' / 'weights' /
                                f'{encoder_type}_{coding_size}_0' / f'{encoder_type}_{coding_size}.ckpt')

        if os.path.exists(f"{training_weight_path}.index"):
            num = int(str(training_weight_path.parents[0]).rsplit("_")[-1])
            training_weight_path = (Path(self.project_path) / 'Training' /
                                    'weights' / f'{encoder_type}_{coding_size}_{num + 1}' /
                                    f'{encoder_type}_{coding_size}.ckpt')
        auto.model.save_weights(training_weight_path)

        return train_model, auto

    def auto_retrain(self, scorer_type='CNN', encoder_type='SAE', coding_size=16, epochs=5,
                     batch_size=256, earlystop=10, verbose=1, gpu='0',
                     scaling_factor=10):

        mat_path = Path(self.project_path) / 'autoencoder_data'
        mat_files = sorted(glob.glob(f'{str(mat_path)}/**/*{scorer_type}_ego*.mat', recursive=True))

        data = loadmat(mat_files[0])
        data = data['animal_d']

        checkpoint_path = Path(self.project_path) / 'Training'

        if encoder_type == 'SAE':
            auto = StackedAE(num_feat=data.shape[1], gpu=gpu,
                             epochs=epochs, batch_size=batch_size, coding_size=coding_size,
                             checkpoint_path=checkpoint_path, training_num=f'{encoder_type}_{coding_size}',
                             earlystop=earlystop, verbose=verbose)

        elif encoder_type == 'VAE':
            auto = VariationalAE(num_feat=data.shape[1], gpu=gpu,
                                 epochs=epochs, batch_size=batch_size, coding_size=coding_size,
                                 checkpoint_path=checkpoint_path, training_num=f'{encoder_type}_{coding_size}',
                                 earlystop=earlystop, scaling_factor=scaling_factor, verbose=verbose)

        weights = sorted(glob.glob(f"{str(checkpoint_path.resolve())}/weights/**/{encoder_type}*.ckpt.index",
                                   recursive=True))
        weights = weights[-1]
        weights = weights.rsplit('.', 1)[0]
        num = int(Path(weights).parts[-2].rsplit('_')[-1])
        auto.model.load_weights(weights)

        for mat_file in mat_files:
            mat_p = Path(mat_file).resolve()
            print(mat_p.stem)
            data = loadmat(mat_file)
            data = data['animal_d']
            train_model = auto.train(data)

        training_model_path = (Path(self.project_path) / 'Training' / 'models' /
                               f'model_{encoder_type}_{coding_size}_{num + 1}')
        auto.model.save(training_model_path)

        training_weight_path = (Path(self.project_path) / 'Training' / 'weights' /
                                f'{encoder_type}_{coding_size}_{num + 1}' / f'{encoder_type}_{coding_size}.ckpt')

        auto.model.save_weights(training_weight_path)

        return train_model, auto

    def predict_w_trained_network(self, scorer_type='CNN', encoder_type='SAE'):

        mat_path = Path(self.project_path) / 'autoencoder_data'
        mat_files = sorted(glob.glob(f'{str(mat_path)}/**/*{scorer_type}_ego*.mat', recursive=True))

        model_path = Path(self.project_path) / 'Training' / 'models'
        model_path = sorted(glob.glob(f"{str(model_path.resolve())}/model*{encoder_type}*", recursive=True))

        model_file = model_path[-1]
        print(model_file)

        auto = k.models.load_model(model_file)

        for file in mat_files:
            a = file[:file.find(f'_{scorer_type}_ego')]
            b = file.rsplit('_', 2)[1]
            filename = f"{a}_{encoder_type}_ego_animal_{b}_data.mat"
            if not os.path.exists(filename):
                print(filename)
                animal_d = loadmat(file)
                animal_d = animal_d['animal_d']
                predicted_d = auto.predict(animal_d)
                # There might be NANs in the data, so you need to take care of them
                predicted_d = pd.DataFrame(predicted_d)
                predicted_d.interpolate(inplace=True)
                predicted_d.fillna(method='bfill', inplace=True)
                predicted_d = predicted_d.to_numpy()
                savemat(filename, {'animal_d': predicted_d})
                del animal_d, predicted_d
