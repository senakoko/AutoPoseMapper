from easydict import EasyDict as eDict


def set_run_parameter(parameters=None):
    if isinstance(parameters, dict):
        parameters = eDict(parameters)
    else:
        parameters = eDict()

    # Names for the processed files
    conv_tracker_name = 'CNN'

    auto_tracker_name = 'SAE'

    animal_key = 'vole_d'

    # Names for the folders that created for the new project
    config_name = 'config_auto.yaml'

    video_path_name = 'videos'

    sleap_data_name = 'sleap_data'

    dlc_data_name = 'dlc_data'

    autoencoder_data_name = 'autoencoder_data'

    id_tracker_data_name = 'id_tracker_data'

    bash_files_name = 'bash_files'

    conv_autoencoder_data_name = 'conv_autoencoder_data'

    # Parameters for SLEAP helper functions

    sleap_track_animal_name = 'sleap_track_animal.sh'

    clean_track_animal_name = 'clean_tracked_animal.sh'

    convert_cleaned_slp = 'convert_cleaned_slp.sh'

    if 'conv_tracker_name' not in parameters.keys():
        parameters.conv_tracker_name = conv_tracker_name

    if 'auto_tracker_name' not in parameters.keys():
        parameters.auto_tracker_name = auto_tracker_name

    if 'animal_key' not in parameters.keys():
        parameters.animal_key = animal_key

    if 'config_name' not in parameters.keys():
        parameters.config_name = config_name

    if 'video_path_name' not in parameters.keys():
        parameters.video_path_name = video_path_name

    if 'sleap_data_name' not in parameters.keys():
        parameters.sleap_data_name = sleap_data_name

    if 'dlc_data_name' not in parameters.keys():
        parameters.dlc_data_name = dlc_data_name

    if 'autoencoder_data_name' not in parameters.keys():
        parameters.autoencoder_data_name = autoencoder_data_name

    if 'id_tracker_data_name' not in parameters.keys():
        parameters.id_tracker_data_name = id_tracker_data_name

    if 'bash_files_name' not in parameters.keys():
        parameters.bash_files_name = bash_files_name

    if 'conv_autoencoder_data_name' not in parameters.keys():
        parameters.conv_autoencoder_data_name = conv_autoencoder_data_name

    if 'sleap_track_animal_name' not in parameters.keys():
        parameters.sleap_track_animal_name = sleap_track_animal_name

    if 'clean_track_animal_name' not in parameters.keys():
        parameters.clean_track_animal_name = clean_track_animal_name

    if 'convert_cleaned_slp.sh' not in parameters.keys():
        parameters.convert_cleaned_slp = convert_cleaned_slp

    return parameters
