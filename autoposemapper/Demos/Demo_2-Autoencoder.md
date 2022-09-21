### DEMO TO USE AUTOENCODER FUNCTIONS

Follow the instructions here to use the autoencoder functions to correct tracking errors.

1. ```
   import sys

    ### Provide the path to the code directory
    sys.path.append('/path_to_where_the_package_is_downloaded_to/AutoPoseMapper/')
   ```
   
2. ```
   from autoposemapper.autoencoder.autoencoder_tools import AutoEncoderHelper
   from autoposemapper.autoencoder.autoencoder_train import AutoTrain
   from autoposemapper.autoencoder import utils as AE_utils
   from autoposemapper.auxiliary_tools import utils as AX_utils
   ```
   
3. ```
   project_path = '/the_path_to_project_folder/'
   ```
   
4. ```
   autoenc = AutoEncoderHelper(project_path)
   ```
   
Egocenter the files

5. ```
   autoenc.egocenter_files(bind_center='midBody', b1='Nose', b2='tailStart', 
                            drop_point=True, which_points=['tailEnd'])
   ```

Train Autoencoder Network

6. ```
   autotrain = AutoTrain(project_path)
   ```
   
The coding size and the number of determines the performance of the network.
Tested with my data the following works well.
For Stacked Autoencoder Network:

- coding_size = 16
- epochs = 100

7. ```
   history, network = autotrain.auto_train_initial(scorer_type='CNN', encoder_type='SAE', 
                                                coding_size=16, epochs=10)
   ```
   
Check the performance of the network

8. ```
   AE_utils.model_loss_plots(history)
   ```
   
9. ```
   AE_utils.model_accuracy_plots(history)
   ```
   
Create Skeleton of egocenter before running this code below

10. ```
    AE_utils.plot_trained_points(network, project_path, frame_number=1500)
    ```
    
If you trained a Variational Autoencoder Network, VAE, you can use the code below

11. ```
    AE_utils.generate_random_animal_w_vae(auto, project_path)
    ```
    
Re-train the network if needed

12. ```
    history, network = autotrain.auto_retrain(scorer_type='CNN', encoder_type='SAE', 
                                          coding_size=16, epochs=10)
    ```
    
Predict with the Trained Network

13. ```
    autotrain.predict_w_trained_network(scorer_type='CNN', encoder_type='SAE')
    ```
    
Re-orient the ego-centered Autoencoder files to original locations

14. ```
    autoenc.reorient_files(encoder_type='SAE')
    ```

Create Pandas' table-style h5 files

15. ```
    autoenc.save_mat_to_h5(encoder_type='SAE')
    ```
    
Check the h5 files

16. ```
    file_path = '/Path_to_an_H5_file/Cohab1_pre_20211025_124326434_SAE_animal_1_data.h5'
    AX_utils.check_pandas_h5(file_path)
    ```
    
Combine H5 files

17. ```
    autoenc.combine_animal_h5_files(encoder_type='SAE')
    ```
    
