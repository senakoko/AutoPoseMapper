import glob

import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import pandas as pd
from autoposemapper.setRunParameters import set_run_parameter


def model_loss_plots(train_model):
    hist_summary = pd.DataFrame(train_model.history)
    hist_summary.plot(y=['loss', 'val_loss'])
    plt.xlabel('Epochs')
    plt.ylabel('RMSE (px)')


def model_accuracy_plots(train_model):
    hist_summary = pd.DataFrame(train_model.history)
    hist_summary.plot(y=['accuracy', 'val_accuracy'])
    plt.xlabel('Epochs')
    plt.ylabel('accuracy %')


def plot_image(image):
    plt.imshow(image, cmap="binary")
    plt.axis("off")


def show_reconstruction(auto, project_path, num_feat=128, batch_size=256,
                        posture_num=5):

    parameters = set_run_parameter()
    test_dir = Path(project_path) / parameters.conv_autoencoder_data_name / 'test/'
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    test_generator = test_datagen.flow_from_directory(test_dir,
                                                      target_size=(num_feat, num_feat),
                                                      batch_size=batch_size,
                                                      class_mode='input')
    reconstructions = auto.model.predict(test_generator[0][0])

    plt.figure(figsize=(posture_num * 20, 40))
    n_in = np.random.randint(0, batch_size, posture_num)

    for image_index, image_value in enumerate(n_in):
        plt.subplot(2, posture_num, 1 + image_index)
        plot_image(test_generator[0][0][image_value])
        plt.subplot(2, posture_num, 1 + posture_num + image_index)
        plot_image(reconstructions[image_value])


def generate_random_animal_w_vae(auto, posture_num=5, coding_size=16):
    codings = tf.random.normal(shape=[posture_num, coding_size])
    images = auto.conv_decoder(codings).numpy()

    # Create images
    plt.figure(figsize=(images.shape[0] * 50, 55))
    for i in range(images.shape[0]):
        plt.subplot(1, images.shape[0], 1 + i)
        plot_image(images[i])
    plt.show()
