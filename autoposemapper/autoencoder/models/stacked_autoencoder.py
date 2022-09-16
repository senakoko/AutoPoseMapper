from tensorflow import keras as k
import tensorflow as tf
import os


class StackedAE(tf.keras.Model):
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
                 validation_split=0.2,
                 verbose=1):
        super(StackedAE, self).__init__()
        if checkpoint_path is None:
            print('Please provide a place to save checkpoints')
            return
        else:
            self.checkpoint_path = checkpoint_path
            self.training_num = training_num
            self.checkpoint_dir = f"{self.checkpoint_path}/training_{self.training_num}/"
            if not os.path.exists(self.checkpoint_dir):
                os.makedirs(self.checkpoint_dir)

        self.gpu = gpu
        if self.gpu is None:
            os.environ["CUDA_VISIBLE_DEVICES"] = '0'
            self.gpu = '0'
        elif isinstance(self.gpu, int):
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu

        self.trainX = trainX
        self.batch_size, self.epochs = batch_size, epochs
        self.num_feat = num_feat
        self.earlystop = earlystop
        self.validation_split = validation_split
        self.verbose = verbose
        self.coding_size = coding_size

        self.model = self._model()

    def _model(self):
        """ Defines the model architecture for the network"""

        # Building the encoder
        input_ = k.layers.Input(shape=[self.num_feat])
        enc = k.layers.Dense(256, activation='selu', name='Encoding1')(input_)
        batch_norm = k.layers.BatchNormalization(name='batchnorm1')(enc)
        drop = k.layers.Dropout(rate=0.2, name='dropout1')(batch_norm)
        enc = k.layers.Dense(128, activation='selu', name='Encoding2')(drop)
        batch_norm = k.layers.BatchNormalization(name='batchnorm2')(enc)
        drop = k.layers.Dropout(rate=0.2, name='dropout2')(batch_norm)
        enc = k.layers.Dense(64, activation='selu', name='Encoding3')(drop)
        batch_norm = k.layers.BatchNormalization(name='batchnorm3')(enc)
        drop = k.layers.Dropout(rate=0.2, name='dropout3')(batch_norm)
        enc = k.layers.Dense(32, activation='selu', name='Encoding4')(drop)
        batch_norm = k.layers.BatchNormalization(name='batchnorm4')(enc)
        drop = k.layers.Dropout(rate=0.2, name='dropout4')(batch_norm)
        enc = k.layers.Dense(self.coding_size, activation='selu', name='Encoding_final')(drop)
        self.encoder = k.Model(inputs=[input_], outputs=[enc])

        # Building the decoder
        decoder_inputs = k.layers.Input(shape=[self.coding_size])
        dec = k.layers.Dense(32, activation='selu', name='Decoding1')(decoder_inputs)
        drop = k.layers.Dropout(rate=0.2, name='dropout5')(dec)
        batch_norm = k.layers.BatchNormalization(name='batchnorm5')(drop)
        dec = k.layers.Dense(64, activation='selu', name='Decoding2')(batch_norm)
        drop = k.layers.Dropout(rate=0.2, name='dropout6')(dec)
        batch_norm = k.layers.BatchNormalization(name='batchnorm6')(drop)
        dec = k.layers.Dense(128, activation='selu', name='Decoding3')(batch_norm)
        drop = k.layers.Dropout(rate=0.2, name='dropout7')(dec)
        batch_norm = k.layers.BatchNormalization(name='batchnorm7')(drop)
        dec = k.layers.Dense(256, activation='selu', name='Decoding4')(batch_norm)
        drop = k.layers.Dropout(rate=0.2, name='dropout8')(dec)
        batch_norm = k.layers.BatchNormalization(name='batchnorm8')(drop)
        output_ = k.layers.Dense(self.num_feat, activation='linear', name='Decoding_final')(batch_norm)
        self.decoder = k.Model(inputs=[decoder_inputs], outputs=[output_])

        codings = self.encoder(input_)
        reconstructions = self.decoder(codings)
        model = k.Model(inputs=[input_], outputs=[reconstructions])

        model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

        return model

    def train(self, trainX=None):
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

        history_sae = self.model.fit(trainX, trainX, epochs=self.epochs,
                                     batch_size=self.batch_size,
                                     validation_split=self.validation_split,
                                     callbacks=[cp_callback, cp_earlystop],
                                     verbose=self.verbose)

        return history_sae
