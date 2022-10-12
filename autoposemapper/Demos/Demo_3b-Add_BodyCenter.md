### DEMO TO ADD THE BODY CENTER¶

Follow the instructions here to fix bad tracking, where the body parts (e.g. nose and tail) of the animal are abnormal.

1. ```
   import sys

    ### Provide the path to the code directory
    sys.path.append('/path_to_where_the_package_is_downloaded_to/AutoPoseMapper/')
   ```
   
2. ```
   from autoposemapper.auxiliary_tools.addBodyCenter import add_body_center
   ```
   
3. ```
   project_path = '/the_path_to_project_folder/'
   ```
   
4. Load the h5 files 
   ```
   h5_files = sorted(glob.glob(f'{project_path}**/*ANT_filtered.h5', recursive=True))
   h5_files
   ```
5. ```
   for file in h5_files:
       h5 = add_body_center(file)
   ```