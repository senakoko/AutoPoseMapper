import pandas as pd
import matplotlib.pyplot as plt
import glob
from pathlib import Path
from scipy.io import loadmat
import yaml
import tensorflow as tf
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


def plot_trained_points(auto, project_path, scorer_type='CNN', frame_number=100):

    parameters = set_run_parameter()
    skeleton_path = Path(project_path) / parameters.skeleton_ego
    mat_path = Path(project_path) / parameters.autoencoder_data_name
    mat_files = sorted(glob.glob(f'{str(mat_path)}/**/*{scorer_type}_ego*.mat', recursive=True))

    data = loadmat(mat_files[0])
    data = data[parameters.animal_key]

    y_predicted = auto.model.predict(data)

    with open(skeleton_path, 'r') as f:
        sk = yaml.load(f, Loader=yaml.FullLoader)
    conn = sk['Skeleton']
    connect = list(conn)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    alpha = 0.6
    for fr in range(frame_number, frame_number+1):
        ax.clear()
        for con in connect:
            ax.plot(data[fr, :][::2][con], data[fr, :][1::2][con], 'b-', alpha=alpha)
            ax.scatter(data[fr, :][::2], data[fr, :][1::2], color='b', s=20, alpha=alpha)
            ax.plot(y_predicted[fr, :][::2][con], y_predicted[fr, :][1::2][con], 'r-', alpha=alpha)
            ax.scatter(y_predicted[fr, :][::2], y_predicted[fr, :][1::2], color='r', s=20, alpha=alpha)
        ax.legend([ax.plot([], [], 'b-')[0], ax.plot([], [], 'r-')[0]], ['True', 'Predicted'], fontsize=24)
    plt.show()


def plot_image(images, connect):
    alpha = 1
    plt.axis('off')
    for con in connect:
        plt.plot(images[::2][con], images[1::2][con], 'b-', alpha=alpha)
        plt.scatter(images[::2], images[1::2], color='b', s=20, alpha=alpha)


def generate_random_animal_w_vae(auto, project_path, posture_num=5, coding_size=16):
    parameters = set_run_parameter()
    codings = tf.random.normal(shape=[posture_num, coding_size])
    images = auto.variational_decoder(codings).numpy()

    skeleton_path = Path(project_path) / parameters.skeleton_ego
    with open(skeleton_path, 'r') as f:
        sk = yaml.load(f, Loader=yaml.FullLoader)
    conn = sk['Skeleton']
    connect = list(conn)

    # Create images
    plt.figure(figsize=(images.shape[0] * 3, 30))
    for i in range(images.shape[0]):
        plt.subplot(images.shape[0], 2, 1 + i)
        plot_image(images[i], connect)
    plt.show()
