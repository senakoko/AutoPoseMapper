from tensorflow import keras as k
import tensorflow as tf
import os
import numpy as np


class StackedCE(tf.keras.Model):
    """
    Performs an encoding on features extracted from the convolutional network

    Parameters
    ----------
    trainX : training data
    num_feat :the number of features from the training dataset
    gpu : which gpu to use
    batch_size : training batch size
    epochs : number of epochs
    checkpoint_path : the path to store the training checkpoint
    training_num : the training instance number
    codings = units for the latent layer
    earlystop: monitors the validation loss and stops the training if there is no improvement
    verbose: Display training information
    validation_split: fraction to split dataset for validation

    """

    def __init__(self, trainX=None,
                 num_feat=42,
                 gpu=None,
                 batch_size=128,
                 epochs=100,
                 coding_size=16,
                 checkpoint_path=None,
                 training_num=1,
                 earlystop=10,
                 generator=True,
                 validation_split=0.2,
                 verbose=1,
                 color_channel_num=3):
        super(StackedCE, self).__init__()
        if checkpoint_path is None:
            print('Please provide a place to save checkpoints')
            return
        else:
            self.checkpoint_path = checkpoint_path
            self.training_num = training_num
            self.checkpoint_dir = f"{self.checkpoint_path}/training_{self.training_num}/"
            if not os.path.exists(self.checkpoint_dir):
                os.makedirs(self.checkpoint_dir)

        if isinstance(gpu, int):
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
            self.gpu = True
        else:
            os.environ["CUDA_VISIBLE DEVICES"] = '0'
            self.gpu = False

        self.trainX = trainX
        self.batch_size, self.epochs = batch_size, epochs
        self.num_feat = num_feat
        self.earlystop = earlystop
        self.validation_split = validation_split
        self.verbose = verbose
        self.coding_size = coding_size
        self.generator = generator
        self.color_channel_num = color_channel_num

        self.model = self._model()

    def _model(self):
        """ Defines the model architecture for the network"""
        if self.color_channel_num == 1:
            chan = 1  # color channel
            kstride = 3  # kernel stride for the deconv layer
        else:
            chan = 3
            kstride = 2

        # Building the encoder
        input_ = k.layers.Input(shape=[self.num_feat, self.num_feat, chan])
        conv = k.layers.Conv2D(16, kernel_size=3, padding='same', activation='selu')(input_)
        batch_norm = k.layers.BatchNormalization()(conv)
        pool = k.layers.MaxPool2D(pool_size=2)(batch_norm)
        conv = k.layers.Conv2D(32, kernel_size=3, padding='same', activation='selu')(pool)
        batch_norm = k.layers.BatchNormalization()(conv)
        pool = k.layers.MaxPool2D(pool_size=2)(batch_norm)
        conv = k.layers.Conv2D(64, kernel_size=3, padding='same', activation='selu')(pool)
        batch_norm = k.layers.BatchNormalization()(conv)
        pool = k.layers.MaxPool2D(pool_size=2)(batch_norm)
        flat = k.layers.Flatten()(pool)
        dense = k.layers.Dense(2048, activation='selu')(flat)
        batch_norm = k.layers.BatchNormalization()(dense)
        drop = k.layers.Dropout(0.2)(batch_norm)
        dense = k.layers.Dense(1024, activation='selu')(drop)
        batch_norm = k.layers.BatchNormalization()(dense)
        drop = k.layers.Dropout(0.2)(batch_norm)
        dense = k.layers.Dense(512, activation='selu')(drop)
        batch_norm = k.layers.BatchNormalization()(dense)
        drop = k.layers.Dropout(0.2)(batch_norm)
        dense = k.layers.Dense(256, activation='selu')(drop)
        batch_norm = k.layers.BatchNormalization()(dense)
        drop = k.layers.Dropout(0.2)(batch_norm)
        dense = k.layers.Dense(128, activation='selu')(drop)
        drop = k.layers.Dropout(0.2)(dense)
        dense = k.layers.Dense(self.coding_size, activation='selu')(drop)
        self.conv_encoder = k.Model(inputs=[input_], outputs=[dense])

        # Building the decoder
        dec_inputs = k.layers.Input(shape=[self.coding_size])
        dense = k.layers.Dense(128, activation='selu')(dec_inputs)
        batch_norm = k.layers.BatchNormalization()(dense)
        drop = k.layers.Dropout(0.2)(batch_norm)
        dense = k.layers.Dense(256, activation='selu')(drop)
        batch_norm = k.layers.BatchNormalization()(dense)
        drop = k.layers.Dropout(0.2)(batch_norm)
        dense = k.layers.Dense(512, activation='selu')(drop)
        batch_norm = k.layers.BatchNormalization()(dense)
        drop = k.layers.Dropout(0.2)(batch_norm)
        dense = k.layers.Dense(1024, activation='selu')(drop)
        batch_norm = k.layers.BatchNormalization()(dense)
        drop = k.layers.Dropout(0.2)(batch_norm)
        dense = k.layers.Dense(2048, activation='selu')(drop)
        batch_norm = k.layers.BatchNormalization()(dense)
        drop = k.layers.Dropout(0.2)(batch_norm)
        dense = k.layers.Dense(int(np.floor(self.num_feat / 8) * np.floor(self.num_feat / 8) * self.coding_size),
                               activation='selu')(drop)
        decoder_inputs = k.layers.Reshape(
            [np.floor(self.num_feat / 8).astype('int'), np.floor(self.num_feat / 8).astype('int'),
             self.coding_size])(dense)
        conv_trans = k.layers.Conv2DTranspose(32, kernel_size=kstride, strides=2, padding='valid',
                                              activation='selu')(decoder_inputs)
        batch_norm = k.layers.BatchNormalization()(conv_trans)
        conv_trans = k.layers.Conv2DTranspose(16, kernel_size=3, strides=2, padding='same', activation='selu')(
            batch_norm)
        batch_norm = k.layers.BatchNormalization()(conv_trans)
        conv_trans = k.layers.Conv2DTranspose(chan, kernel_size=3, strides=2, padding='same', activation='sigmoid')(
            batch_norm)
        if chan == 1:
            conv_reshape = k.layers.Reshape([self.num_feat, self.num_feat])(conv_trans)
        else:
            conv_reshape = k.layers.Reshape([self.num_feat, self.num_feat, chan])(conv_trans)
        self.conv_decoder = k.Model(inputs=[dec_inputs], outputs=[conv_reshape])

        codings = self.conv_encoder(input_)
        reconstructions = self.conv_decoder(codings)
        model = k.Model(inputs=[input_], outputs=[reconstructions])

        model.compile(loss='binary_crossentropy', optimizer=k.optimizers.Adam(lr=0.001), metrics=['accuracy'])

        return model

    def train(self, trainX=None, valX=None):
        """ Train the model to make predictions"""

        if trainX is None:
            try:
                trainX = self.trainX
            except trainX.NotProvided:
                raise ValueError('Please provide trainX')

        #  Create a callback that saves the model's weights
        check_path = self.checkpoint_dir + 'cp-{epoch:04d}.ckpt'
        cp_callback = k.callbacks.ModelCheckpoint(filepath=check_path, verbose=1,
                                                  save_weights_only=True,
                                                  save_best_only=True)

        cp_earlystop = k.callbacks.EarlyStopping(patience=self.earlystop,
                                                 restore_best_weights=True)

        # Save the weights using the `checkpoint_path` format
        self.model.save_weights(check_path.format(epoch=0))

        if not self.generator:
            history_sae = self.model.fit(trainX, trainX, epochs=self.epochs,
                                         batch_size=self.batch_size,
                                         validation_split=self.validation_split,
                                         callbacks=[cp_callback, cp_earlystop],
                                         verbose=self.verbose)
        else:
            history_sae = self.model.fit(trainX, epochs=self.epochs,
                                         batch_size=self.batch_size,
                                         validation_data=valX,
                                         callbacks=[cp_callback, cp_earlystop],
                                         verbose=self.verbose)

        return history_sae
