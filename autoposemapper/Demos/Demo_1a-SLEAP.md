### DEMO TO USE SLEAP HELPER FUNCTIONS

Follow the instructions here to use the SLEAP helper tools.   
It assumes you already have used SLEAP to train a network.

1. ```
   import sys

    ### Provide the path to the code directory
    sys.path.append('/path_to_where_the_package_is_downloaded_to/AutoPoseMapper/')
   ```
   
2. ```
   import glob
   from autoposemapper.sleap_tools.sleap_tools import SleapHelper
   from autoposemapper.auxiliary_tools import utils
   ```
   
3. ```
   project_path = '/the_path_to_project_folder/'
   ```
   
4. ```
   sleap = SleapHelper(project_path)
   ```
   
Create a bash script to use SLEAP to track videos

5. ```
   sleap.track_videos_sleap(model_path='path_to_trained_network', video_type='.mp4')
   ```
Create a bash script to use SLEAP to clean tracked videos

6. ```
   sleap.clean_tracked_files()
   ```
Create a bash script to use SLEAP to make the h5 files from the cleaned sleap files

7. ```
   sleap.convert_sleap_2_h5()
   ```
Convert SLEAP's h5 files to Pandas' table-style h5

8. ```
   sleap.convert_sleap_h5_2_pandas_h5()
   ```
Check the H5 files

9. ```
   file_path = '/Path_to_an_H5_File/Cohab1_pre_20211025_124326434_CNN.h5'
   h5 = utils.check_pandas_h5(file_path)
   h5
   ```