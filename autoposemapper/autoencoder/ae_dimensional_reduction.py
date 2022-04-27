import os
import glob
from pathlib import Path
import pandas as pd
from autoposemapper.autoencoder.models.stacked_autoencoder import StackedAE
from autoposemapper.autoencoder.models.variational_autoencoder import VariationalAE


class AutoTrainDimRed:
    def __init__(self, project_path, body_info_path):
        self.project_path = project_path
        self.body_info_path = body_info_path

    def auto_train_dimred(self, encoder_type='SAE', coding_size=8, epochs=5,
                          batch_size=256, earlystop=10, verbose=1, gpu='0',
                          scaling_factor=10):

        data_path = Path(self.body_info_path)
        data_files = sorted(glob.glob(f'{str(data_path)}/*.h5', recursive=True))

        data = pd.read_hdf(data_files[0])

        checkpoint_path = Path(self.project_path) / 'Training'
        if encoder_type == 'SAE':
            auto = StackedAE(num_feat=data.shape[1], gpu=gpu,
                             epochs=epochs, batch_size=batch_size, coding_size=coding_size,
                             checkpoint_path=checkpoint_path, training_num=f'DimRed_{encoder_type}_{coding_size}',
                             earlystop=earlystop, verbose=verbose)

        elif encoder_type == 'VAE':
            auto = VariationalAE(num_feat=data.shape[1], gpu=gpu,
                                 epochs=epochs, batch_size=batch_size, coding_size=coding_size,
                                 checkpoint_path=checkpoint_path, training_num=f'DimRed_{encoder_type}_{coding_size}',
                                 earlystop=earlystop, scaling_factor=scaling_factor, verbose=verbose)

        for data_file in data_files:
            data_file_p = Path(data_file).resolve()
            print(data_file_p.stem)
            data = pd.read_hdf(data_file)
            data.interpolate(inplace=True)
            data.fillna(method='bfill', inplace=True)
            data.fillna(method='ffill', inplace=True)
            trainX = data.values
            train_model = auto.train(trainX)

        training_model_path = (Path(self.project_path) / 'Training' / 'dimred_models' /
                               f'model_{encoder_type}_{coding_size}_0')

        if training_model_path.exists():
            num = int(str(training_model_path).rsplit("_")[-1])
            training_model_path = (Path(self.project_path) / 'Training' /
                                   'dimred_models' / f'model_{encoder_type}_{coding_size}_{num + 1}')
        auto.model.save(training_model_path)

        training_weight_path = (Path(self.project_path) / 'Training' / 'dimred_weights' /
                                f'{encoder_type}_{coding_size}_0' / f'{encoder_type}_{coding_size}.ckpt')

        if os.path.exists(f"{training_weight_path}.index"):
            num = int(str(training_weight_path.parents[0]).rsplit("_")[-1])
            training_weight_path = (Path(self.project_path) / 'Training' /
                                    'dimred_weights' / f'{encoder_type}_{coding_size}_{num + 1}' /
                                    f'{encoder_type}_{coding_size}.ckpt')
        auto.model.save_weights(training_weight_path)

        return train_model, auto

    def auto_retrain_dimred(self, encoder_type='SAE', coding_size=8, epochs=5,
                            batch_size=256, earlystop=10, verbose=1, gpu='0',
                            scaling_factor=10):

        data_path = Path(self.body_info_path)
        data_files = sorted(glob.glob(f'{str(data_path)}/*.h5', recursive=True))

        data = pd.read_hdf(data_files[0])

        checkpoint_path = Path(self.project_path) / 'Training'

        if encoder_type == 'SAE':
            auto = StackedAE(num_feat=data.shape[1], gpu=gpu,
                             epochs=epochs, batch_size=batch_size, coding_size=coding_size,
                             checkpoint_path=checkpoint_path, training_num=f'DimRed_{encoder_type}_{coding_size}',
                             earlystop=earlystop, verbose=verbose)

        elif encoder_type == 'VAE':
            auto = VariationalAE(num_feat=data.shape[1], gpu=gpu,
                                 epochs=epochs, batch_size=batch_size, coding_size=coding_size,
                                 checkpoint_path=checkpoint_path, training_num=f'DimRed_{encoder_type}_{coding_size}',
                                 earlystop=earlystop, scaling_factor=scaling_factor, verbose=verbose)

        weights = sorted(glob.glob(f"{str(checkpoint_path.resolve())}/dimred_weights/**/{encoder_type}*.ckpt.index",
                                   recursive=True))
        weights = weights[-1]
        weights = weights.rsplit('.', 1)[0]
        num = int(Path(weights).parts[-2].rsplit('_')[-1])
        auto.model.load_weights(weights)

        for data_file in data_files:
            data_file_p = Path(data_file).resolve()
            print(data_file_p.stem)
            data = pd.read_hdf(data_file)
            data.interpolate(inplace=True)
            data.fillna(method='bfill', inplace=True)
            data.fillna(method='ffill', inplace=True)
            trainX = data.values
            train_model = auto.train(trainX)

        training_model_path = (Path(self.project_path) / 'Training' / 'dimred_models' /
                               f'model_{encoder_type}_{coding_size}_{num + 1}')
        auto.model.save(training_model_path)

        training_weight_path = (Path(self.project_path) / 'Training' / 'dimred_weights' /
                                f'{encoder_type}_{coding_size}_{num + 1}' / f'{encoder_type}_{coding_size}.ckpt')

        auto.model.save_weights(training_weight_path)

        return train_model, auto

    def reduce_dimensions(self, encoder_type='SAE', coding_size=8, epochs=5,
                          batch_size=256, earlystop=10, verbose=1, gpu='0',
                          scaling_factor=10):

        data_path = Path(self.body_info_path)
        data_files = sorted(glob.glob(f'{str(data_path)}/*.h5', recursive=True))

        destination_path = Path(self.project_path) / f'Encoded_{Path(self.body_info_path).stem}' / 'Encoded'
        destination_path = destination_path.resolve()

        if not destination_path.exists():
            destination_path.mkdir(parents=True)
            print(f'{destination_path.stem} folder made')

        data = pd.read_hdf(data_files[0])

        checkpoint_path = Path(self.project_path) / 'Training'

        if encoder_type == 'SAE':
            auto = StackedAE(num_feat=data.shape[1], gpu=gpu,
                             epochs=epochs, batch_size=batch_size, coding_size=coding_size,
                             checkpoint_path=checkpoint_path, training_num=f'DimRed_{encoder_type}_{coding_size}',
                             earlystop=earlystop, verbose=verbose)

        elif encoder_type == 'VAE':
            auto = VariationalAE(num_feat=data.shape[1], gpu=gpu,
                                 epochs=epochs, batch_size=batch_size, coding_size=coding_size,
                                 checkpoint_path=checkpoint_path, training_num=f'DimRed_{encoder_type}_{coding_size}',
                                 earlystop=earlystop, scaling_factor=scaling_factor, verbose=verbose)

        weights = sorted(glob.glob(f"{str(checkpoint_path.resolve())}/dimred_weights/**/{encoder_type}*.ckpt.index",
                                   recursive=True))
        weights = weights[-1]
        weights = weights.rsplit('.', 1)[0]
        auto.model.load_weights(weights)

        for file in data_files:
            file_name = f"{str(destination_path)}/{Path(file).name}"
            if os.path.exists(file_name):
                continue
            data = pd.read_hdf(file)
            data.interpolate(inplace=True)
            data.fillna(method='bfill', inplace=True)
            data.fillna(method='ffill', inplace=True)
            trainX = data.values
            encoded_data = auto.encoder.predict(trainX)
            encoded_df = pd.DataFrame(encoded_data)
            with pd.HDFStore(f'{file_name}') as store:
                store.put('encoded_features', encoded_df)
            del encoded_data, encoded_df
            print('Done Analyzing', file_name)

        return destination_path
