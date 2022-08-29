### DEMO TO USE ID Tracker.AI FUNCTIONS

Follow the instructions here to use ID tracker functions. 
It assumes you created the trajectories.npy files with ID Tracker and copied them to the respective folders

1. ```
   import sys

    ### Provide the path to the code directory
    sys.path.append('/path_to_where_the_package_is_downloaded_to/AutoPoseMapper/')
   ```
   
2. ```
   from autoposemapper.id_tracker_tools import IdTracker
   from autoposemapper.auxiliary_tools import utils as AX_utils
   ```
   
3. ```
   project_path = '/the_path_to_project_folder/'
   ```
   
4. ```
   idtracker = IdTracker(project_path)
   ```
   
Make h5 files from the numpy trajectories

5. ```
   idtracker.make_h5s_id_trajectories()
   ```
   
Check h5 files

6. ```
   file_path = '/Path_to_ID_Tracker_H5_file/Cohab1_pre_20211025_124326434_idtraj.h5'
   h5 = AX_utils.check_pandas_h5(file_path)
   h5
   ```
   
#### Create new centroid based on IDTracker and Autoencoder
Path to the h5 files created by the autoencoder. You want the parent path

7. ```
   auto_source_path = '/Path_to_the_autoencoder_data/'
   idtracker.create_csi_centroid(auto_source_path=auto_source_path)
   ```
   
#### Swap tracked points based on labels

First create the labels based on the id tracker tracking
8. ```
   idtracker.create_labels_based_on_csi(auto_source_path=auto_source_path)
   ```
Swap the tracks based on the id tracker tracking
   
9. ```
   idtracker.swap_tracks_based_on_labels(auto_source_path=auto_source_path)
   ```
   
Check the swapped tracked points

10. ```
    file_path2 = '/Path_to_Swapped_H5_file/Cohab1_pre_20211025_124326434_CSI_filtered.h5'
    h5 = AX_utils.check_pandas_h5(file_path2)
    h5
    ```
    
Fix Bad Areas in H5 files

11. ```
    idtracker.fix_bad_areas_h5()
    ```
    
Create Individual Animals from Multi-Animal Dataset

12. ```
    idtracker.create_ind_from_multi(encoder_type='CSI_Area')
    ```
    
