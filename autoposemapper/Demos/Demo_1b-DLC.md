### DEMO TO USE DeepLabCut HELPER FUNCTIONS

Follow the instructions here to use the DLC helper tools.   
It assumes you already have used DLC to train a network and labeled files

1. ```
   import sys

    ### Provide the path to the code directory
    sys.path.append('/path_to_where_the_package_is_downloaded_to/AutoPoseMapper/')
   ```
   
2. ```
   import glob
   from autoposemapper.deeplabcut_tools.dlc_tools import DlcHelper
   ```
   
3. ```
   project_path = '/the_path_to_project_folder/'
   ```
   
4. ```
   dlc = DlcHelper(project_path)
   ```
   
Copy DLC files to DLC folder under the project

5. ```
   dlc.copy_dlc_files_DD('/Users/senaagezo/Downloads/')
   ```
Copy DLC h5 files from the DLC folder to the Autoencoder Folder

6. ```
   dlc.copy_dlc_files_AE()
   ```

