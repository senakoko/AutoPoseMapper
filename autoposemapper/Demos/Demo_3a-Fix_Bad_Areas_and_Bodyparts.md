### DEMO TO FIX BAD AREAS and BODY PARTS IN TRACKED POINTS

Follow the instructions here to fix bad tracking, where the body parts (e.g. nose and tail) of the animal are abnormal.

1. ```
   import sys

    ### Provide the path to the code directory
    sys.path.append('/path_to_where_the_package_is_downloaded_to/AutoPoseMapper/')
   ```
   
2. ```
   from autoposemapper.auxiliary_tools.fixBadArea_Bodyparts import fix_bad_area_and_bodyparts
   ```
   
3. ```
   project_path = '/the_path_to_project_folder/'
   ```
   
4. Load the h5 files 
   ```
   h5_files = sorted(glob.glob(f'{project_path}**/*CNN_SAE.h5', recursive=True))
   files = []
   for file in h5_files:
       if re.search('labeled', file):
          continue
       files.append(file)
   files
   ```
5. ```
   for file in files:
       fix_bad_area_and_bodyparts(file, mad_multiplier=2.75, save_csv=True)
   ```