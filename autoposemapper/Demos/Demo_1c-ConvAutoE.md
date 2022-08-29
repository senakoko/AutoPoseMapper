### DEMO TO USE CONVOLUTIONAL AUTOENCODER FUNCTIONS

Follow the instructions here to use the convolutional autoencoder tools.

1. ```
   import sys

    ### Provide the path to the code directory
    sys.path.append('/path_to_where_the_package_is_downloaded_to/AutoPoseMapper/')
   ```
   
2. ```
   import glob
   from autoposemapper.convolutional_autoencoder.frame_tools import FrameTools
   from autoposemapper.convolutional_autoencoder.conv_autoencoder_train import AutoTrain
   from autoposemapper.convolutional_autoencoder import utils
   ```
   
3. ```
   project_path = '/the_path_to_project_folder/'
   ```
   
4. ```
   frame_tools = FrameTools(project_path)
   ```
   
Extract frames to train Convolutional Autoencoder

5. ```
   frame_tools.extract_frames_conv(numframes2pick=500, userfeedback=False, algo='uniform', 
   name_prefix='extracted', opencv=True)
   ```
   
Create Training and Test Datasets

6. ```
   frame_tools.create_train_test_datasets(train_fraction=0.8)
   ```
   
### Train the Network

Initialize the convolutional autoencoder 
7. ```
   autotrain = AutoTrain(project_path)
   ```
   
Train the convolutional autoencoder network

8. ```
   history, network = autotrain.auto_train_initial(num_feat=128, encoder_type='VAE', 
   batch_size=64,coding_size=16, epochs=5)
   ```
   
Retrain the convolutional autoencoder network if needed

9. ```
   history, network = autotrain.auto_retrain(num_feat=128, encoder_type='VAE', 
   batch_size=64, coding_size=16, epochs=2)
   ```

Plot the loss history

10. ```
    utils.model_loss_plots(history)
    ```

Show the reconstructions from actual image to check the performance of the network

11. ```
    utils.show_reconstruction(network, project_path, batch_size=64)
    ```
    
Generate random (fake) images based on the trained network. Check how similar they are
to actual images

12. ```
    utils.generate_random_animal_w_vae(network, coding_size=16)
    ```

Reduce the dimensions of the frames from the videos
13. ```
    autotrain.reduce_dimensions(encoder_type='VAE', coding_size=16, 
    batch_size=64, scaling_factor=128. * 128.)
    ```