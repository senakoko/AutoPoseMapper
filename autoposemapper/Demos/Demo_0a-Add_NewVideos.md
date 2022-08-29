### DEMO TO ADD NEW VIDEOS TO THE PROJECT FOLDER

Follow the instructions here to add new videos to an existing project folder.

1. ```
   import sys
   ### Provide the path to the code directory
   sys.path.append('/path_to_where_the_package_is_downloaded_to/AutoPoseMapper/')
   ```
   
2. ```
   from autoposemapper.create_project.add_newvideos import add_new_videos
   ```
   
3. ```
   project_path = '/the_path_to_project_folder/'
   ```
   
4. ```
   add_new_videos(project_path, '/path_to_the_videos/', copy_videos=False, sleapordlc='sleap')
   ```
   
