import os
import glob
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from autoposemapper.convolutional_autoencoder.models.stacked_convolutional_autoencoder import StackedCE
from autoposemapper.convolutional_autoencoder.models.variational_convolutional_autoencoder import VariationalCE
from autoposemapper.convolutional_autoencoder.models.big_variational_convolutional_autoencoder import BigVariationalCE
from autoposemapper.setRunParameters import set_run_parameter


class AutoTrain:
    def __init__(self, project_path, parameters=None):
        self.project_path = project_path
        self.parameters = parameters

        if self.parameters is None:
            self.parameters = set_run_parameter()

    def auto_train_initial(self, num_feat=128, encoder_type='VAE', coding_size=8, epochs=5,
                           batch_size=256, earlystop=10, verbose=1, gpu='0'):

        num_feat_list = [2 ** i for i in range(3, 10)]
        if num_feat not in num_feat_list:
            print(f'the num_feat has to be one of this {num_feat_list}')
            return

        scaling_factor = float(num_feat) * float(num_feat)

        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

        train_dir = Path(self.project_path) / self.parameters.conv_autoencoder_data_name / 'train/'
        test_dir = Path(self.project_path) / self.parameters.conv_autoencoder_data_name / 'test/'

        checkpoint_path = Path(self.project_path) / 'Training'
        if encoder_type == 'SAE' and num_feat <= 128:
            auto = StackedCE(num_feat=num_feat, gpu=gpu,
                             epochs=epochs, batch_size=batch_size, coding_size=coding_size,
                             checkpoint_path=checkpoint_path, training_num=f'conv_{encoder_type}_{coding_size}',
                             earlystop=earlystop, verbose=verbose)

        elif encoder_type == 'VAE' and num_feat <= 128:
            auto = VariationalCE(num_feat=num_feat, gpu=gpu,
                                 epochs=epochs, batch_size=batch_size, coding_size=coding_size,
                                 checkpoint_path=checkpoint_path, training_num=f'conv_{encoder_type}_{coding_size}',
                                 earlystop=earlystop, scaling_factor=scaling_factor, verbose=verbose)

        elif encoder_type == 'VAE' and num_feat >= 256:
            auto = BigVariationalCE(num_feat=num_feat, gpu=gpu,
                                    epochs=epochs, batch_size=batch_size, coding_size=coding_size,
                                    checkpoint_path=checkpoint_path, training_num=f'conv_{encoder_type}_{coding_size}',
                                    earlystop=earlystop, scaling_factor=scaling_factor, verbose=verbose)

        train_datagen = ImageDataGenerator(rescale=1. / 255)
        test_datagen = ImageDataGenerator(rescale=1. / 255)

        train_generator = train_datagen.flow_from_directory(train_dir,
                                                            target_size=(num_feat, num_feat),
                                                            batch_size=batch_size,
                                                            class_mode='input')
        test_generator = test_datagen.flow_from_directory(test_dir,
                                                          target_size=(num_feat, num_feat),
                                                          batch_size=batch_size,
                                                          class_mode='input')

        train_model = auto.train(train_generator, test_generator)

        training_model_path = (Path(self.project_path) / 'Training' / 'conv_models' /
                               f'model_{encoder_type}_{coding_size}_0')

        if training_model_path.exists():
            num = int(str(training_model_path).rsplit("_")[-1])
            training_model_path = (Path(self.project_path) / 'Training' /
                                   'conv_models' / f'model_{encoder_type}_{coding_size}_{num + 1}')
        auto.model.save(training_model_path)

        training_weight_path = (Path(self.project_path) / 'Training' / 'conv_weights' /
                                f'{encoder_type}_{coding_size}_0' / f'{encoder_type}_{coding_size}.ckpt')

        if os.path.exists(f"{training_weight_path}.index"):
            num = int(str(training_weight_path.parents[0]).rsplit("_")[-1])
            training_weight_path = (Path(self.project_path) / 'Training' /
                                    'conv_weights' / f'{encoder_type}_{coding_size}_{num + 1}' /
                                    f'{encoder_type}_{coding_size}.ckpt')
        auto.model.save_weights(training_weight_path)

        return train_model, auto

    def auto_retrain(self, num_feat=128, encoder_type='VAE', coding_size=16, epochs=5,
                     batch_size=256, earlystop=10, verbose=1, gpu='0'):

        num_feat_list = [2 ** i for i in range(3, 10)]
        if num_feat not in num_feat_list:
            print(f'the num_feat has to be one of this {num_feat_list}')
            return

        scaling_factor = float(num_feat) * float(num_feat)

        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

        train_dir = Path(self.project_path) / self.parameters.conv_autoencoder_data_name / 'train/'
        test_dir = Path(self.project_path) / self.parameters.conv_autoencoder_data_name / 'test/'

        checkpoint_path = Path(self.project_path) / 'Training'
        if encoder_type == 'SAE' and num_feat <= 128:
            auto = StackedCE(num_feat=num_feat, gpu=gpu,
                             epochs=epochs, batch_size=batch_size, coding_size=coding_size,
                             checkpoint_path=checkpoint_path, training_num=f'conv_{encoder_type}_{coding_size}',
                             earlystop=earlystop, verbose=verbose)

        elif encoder_type == 'VAE' and num_feat <= 128:
            auto = VariationalCE(num_feat=num_feat, gpu=gpu,
                                 epochs=epochs, batch_size=batch_size, coding_size=coding_size,
                                 checkpoint_path=checkpoint_path, training_num=f'conv_{encoder_type}_{coding_size}',
                                 earlystop=earlystop, scaling_factor=scaling_factor, verbose=verbose)

        elif encoder_type == 'VAE' and num_feat >= 256:
            auto = BigVariationalCE(num_feat=num_feat, gpu=gpu,
                                    epochs=epochs, batch_size=batch_size, coding_size=coding_size,
                                    checkpoint_path=checkpoint_path, training_num=f'conv_{encoder_type}_{coding_size}',
                                    earlystop=earlystop, scaling_factor=scaling_factor, verbose=verbose)

        weights = sorted(glob.glob(f"{str(checkpoint_path.resolve())}/conv_weights/**/{encoder_type}*.ckpt.index",
                                   recursive=True))
        weights = weights[-1]
        weights = weights.rsplit('.', 1)[0]
        num = int(Path(weights).parts[-2].rsplit('_')[-1])
        auto.model.load_weights(weights)

        train_datagen = ImageDataGenerator(rescale=1. / 255)
        test_datagen = ImageDataGenerator(rescale=1. / 255)

        train_generator = train_datagen.flow_from_directory(train_dir,
                                                            target_size=(num_feat, num_feat),
                                                            batch_size=batch_size,
                                                            class_mode='input')
        test_generator = test_datagen.flow_from_directory(test_dir,
                                                          target_size=(num_feat, num_feat),
                                                          batch_size=batch_size,
                                                          class_mode='input')

        train_model = auto.train(train_generator, test_generator)

        training_model_path = (Path(self.project_path) / 'Training' / 'conv_models' /
                               f'model_{encoder_type}_{coding_size}_{num + 1}')
        auto.model.save(training_model_path)

        training_weight_path = (Path(self.project_path) / 'Training' / 'conv_weights' /
                                f'{encoder_type}_{coding_size}_{num + 1}' / f'{encoder_type}_{coding_size}.ckpt')

        auto.model.save_weights(training_weight_path)

        return train_model, auto

    def reduce_dimensions(self, num_feat=128, encoder_type='VAE', coding_size=8, epochs=5,
                          batch_size=256, earlystop=10, verbose=1, gpu='0', video_type='.mp4'):

        num_feat_list = [2 ** i for i in range(3, 10)]
        if num_feat not in num_feat_list:
            print(f'the num_feat has to be one of this {num_feat_list}')
            return

        scaling_factor = float(num_feat) * float(num_feat)

        data_path = Path(self.project_path) / self.parameters.video_path_name
        data_files = sorted(glob.glob(f'{str(data_path)}/**/*{video_type}', recursive=True))

        destination_path = Path(self.project_path) / f'Encoded_Conv_{encoder_type}_{coding_size}' / 'Encoded'
        destination_path = destination_path.resolve()

        if not destination_path.exists():
            destination_path.mkdir(parents=True)
            print(f'{destination_path.stem} folder made')

        checkpoint_path = Path(self.project_path) / 'Training'

        if encoder_type == 'SAE' and num_feat <= 128:
            auto = StackedCE(num_feat=num_feat, gpu=gpu,
                             epochs=epochs, batch_size=batch_size, coding_size=coding_size,
                             checkpoint_path=checkpoint_path, training_num=f'conv_{encoder_type}_{coding_size}',
                             earlystop=earlystop, verbose=verbose)

        elif encoder_type == 'VAE' and num_feat <= 128:
            auto = VariationalCE(num_feat=num_feat, gpu=gpu,
                                 epochs=epochs, batch_size=batch_size, coding_size=coding_size,
                                 checkpoint_path=checkpoint_path, training_num=f'conv_{encoder_type}_{coding_size}',
                                 earlystop=earlystop, scaling_factor=scaling_factor, verbose=verbose)

        elif encoder_type == 'VAE' and num_feat >= 256:
            auto = BigVariationalCE(num_feat=num_feat, gpu=gpu,
                                    epochs=epochs, batch_size=batch_size, coding_size=coding_size,
                                    checkpoint_path=checkpoint_path, training_num=f'conv_{encoder_type}_{coding_size}',
                                    earlystop=earlystop, scaling_factor=scaling_factor, verbose=verbose)

        weights = sorted(glob.glob(f"{str(checkpoint_path.resolve())}/conv_weights/**/{encoder_type}*.ckpt.index",
                                   recursive=True))
        weights = weights[-1]
        weights = weights.rsplit('.', 1)[0]
        auto.model.load_weights(weights)

        for file in data_files:
            # Number of units to decompose the image. Based on units in latent layer of the model
            num_units = coding_size
            # The variational encoder outputs 3 codings - mean[0], log variances[1], random codings[2].
            # The numbers specify which codings you want to use
            # For the stacked encoder, specify '0' because only one coding is generated
            coding_to_pick = 0
            cap = cv2.VideoCapture(file)
            vid_name = Path(file).stem
            file_name = f"{str(destination_path)}/{vid_name}.h5"
            if os.path.exists(file_name):
                continue
            else:
                print('Analyzing', vid_name)
                ret = True
                encoded_features = np.zeros((int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), num_units))
                counter = 0
                dim = (num_feat, num_feat)
                while ret:
                    ret, frame = cap.read()
                    if ret:
                        # the network expects a 4D tensor so this is to reshape the image from 3D to 4D
                        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
                        frame_reshaped = frame[np.newaxis, :]
                        frame_features = auto.conv_encoder.predict(frame_reshaped)
                        encoded_features[counter] = frame_features[coding_to_pick].flatten()
                    counter += 1
                encoded_features_df = pd.DataFrame(encoded_features)
                with pd.HDFStore(f'{destination_path}/{vid_name}.h5') as store:
                    store.put('encoded_features', encoded_features_df)

                del encoded_features, encoded_features_df
                print('Done Analyzing', vid_name)

        return destination_path
