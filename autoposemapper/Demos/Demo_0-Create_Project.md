# Create New Project

Follow the instructions here to create the project folder.
It is easier to have everything centralized under one project folder. 

### Note
Make sure you have activated the conda environment that you created following the README.md instructions.

1. ```
   import sys
   ### Provide the path to the code directory
   sys.path.append('/path_to_where_the_package_is_downloaded_to/AutoPoseMapper/')
   ```
2. ```
   from autoposemapper.create_project.new_project import create_new_project
   ```
3. ```
   project_path = create_new_project(project='AutoTest', experimenter='Sena', 
                                  videos=['/path_to_the_video_file/Cohab1_pre_20211025_124326434.mp4'], 
                                  working_directory='/path_to_the_working_directory/', 
                                  video_type='.mp4', 
                                  sleap_or_dlc_or_conv='dlc',
                                  copy_videos=False)
   ```
   
4. ```
   project_path
   ```
### Need to do this if you are working with the Tracked Points 
1. Copy all the skeleton files under the skeleton folder to your new project directory. 
Put them under a folder titled  **Skeletons**.
Also, you will need to modify the skeleton based on how you want the points plotted.

### Note
If you are running the code on Windows, and you get an error due to mklink, which is used to create a symbolic link. 
First delete the project and create a new one. However, set 
```
copy_videos=True.
```

 
 