### Make Videos of Tracked Points

Follow the instructions here to make videos with tracked points from either SLEAP, 
the autoencoder or the ID Tracker.

1. ```
   import sys

    ### Provide the path to the code directory
    sys.path.append('/path_to_where_the_package_is_downloaded_to/AutoPoseMapper/')
   ```
   
2. ```
   from autoposemapper.id_tracker_tools import IdTracker
   from autoposemapper.auxiliary_tools.make_videos import MakeVideos
   from autoposemapper.auxiliary_tools.downSampleVideos import down_sample_video
   ```
   
3. ```
   project_path = '/the_path_to_project_folder/'
   ```
   
Make Video with SLEAP Points

4. ```
   mvid = MakeVideos(project_path)
   mvid.make_tracked_videos(video_loc='/Users/senaagezo/Downloads/AutoTest-Sena-2022-04-17/videos/',
                        skeleton_path='/Users/senaagezo/Downloads/AutoTest-Sena-2022-04-17/skeleton_unfiltered.yaml',
                         subset=True, start_time=0, end_time=(1,0),post_name='CNN', tracker='CNN')
   
   ```

Make Video with Autoencoder Points

5. ```
   mvid.make_tracked_videos(video_loc='/Users/senaagezo/Downloads/AutoTest-Sena-2022-04-17/videos/',
                        skeleton_path='/Users/senaagezo/Downloads/AutoTest-Sena-2022-04-17/skeleton.yaml',
                         subset=True, start_time=0, end_time=(1,0),post_name='CNN_SAE', tracker='CNN_SAE')
   ```
   
Make Video with ID Tracker Points

6. ```
   idtracker = IdTracker(project_path)
   idtracker.make_id_tracker_movies(video_loc='/Users/senaagezo/Downloads/AutoTest-Sena-2022-04-17/videos/', subset=True,
                          start_time=(0), end_time=(1,0), dot_size=10, tracker='_idtraj')
   ```
   
Make Video with Both Autoencoder and ID Tracker Points

7. ```
   mvid.make_tracked_videos_w_idt(video_loc='/Users/senaagezo/Downloads/AutoTest-Sena-2022-04-17/videos/',
                               skeleton_path='/Users/senaagezo/Downloads/AutoTest-Sena-2022-04-17/skeleton.yaml',
                               subset=True, start_time=0, end_time=(1,0),post_name='CNN_SAE_IDT', tracker='CNN_SAE')
   ```
   
Down-sample videos if you wanted to do so before Tracking

8. ```
   down_sample_video(
    '/Users/senaagezo/Downloads/AutoTest-Sena-2022-04-17/videos/Cohab1_pre_20211025_124326434/Cohab1_pre_20211025_124326434.mp4',
    start_time=0, end_time=(1,0), scale_factor=0.5, subset=True)
   ```