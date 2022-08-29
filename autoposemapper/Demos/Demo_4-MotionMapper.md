### DEMO TO USE MotionMapper FUNCTIONS

Follow the instructions here to use MotionMapper Functions

1. ```
   import sys
   
   ### Provide the path to the code directory
   sys.path.append('/path_to_where_the_package_is_downloaded_to/AutoPoseMapper/')
   ```
   
2. ```
   from autoposemapper.motion_mapper_tools.calculateBodyInfo import CalculateBodyInfo
   from autoposemapper.motion_mapper_tools.checkPCA import check_pca
   from autoposemapper.autoencoder.ae_dimensional_reduction import AutoTrainDimRed
   from autoposemapper.motion_mapper_tools.extractProjections import extract_projections, check_extracted_projections
   from autoposemapper.motion_mapper_tools.runMotionMapper import run_motion_mapper
   from autoposemapper.motion_mapper_tools.createBradyVideos import create_brady_videos, center_video
   from autoposemapper.autoencoder import utils as AE_utils
   ```
   
3. ```
   project_path = '/the_path_to_project_folder/'
   ```

Calculate Euclidean Distances between points

4. ```
   calbody = CalculateBodyInfo(project_path)
   ```
   
5. ```
   body_path = calbody.calculate_body_info(calculation_type='Euc_Dist', 
   encoder_type='SAE')
   ```

Run a PCA on the Euclidean Distances between points
6. ```
   check_pca(body_path)
   ```
   
### Train Autoencoder Network

7. ```
   autotrain = AutoTrainDimRed(project_path, body_path)
   history, network = autotrain.auto_train_dimred(coding_size=8)

   
Check the performance of the network

8. ```
   autoencoder.utils.model_loss_plots(history)
   ```
   
Reduce the Dimensions

9. ```
   encoded_path = autotrain.reduce_dimensions()
   encoded_path
   ```
   
Extract Projections

10. ```
    extract_projections(encoded_path)
    check_extracted_projections(encoded_path)
    ```
    
## Run MotionMapper

Runt the motion mapper pipeline
11. ```
    run_motion_mapper(encoded_path,minF=0.5,maxF=15,perplexity=32,tSNE_method='barnes_hut',
                  samplingFreq=30,numPeriods=50,numProcessors=4,useGPU=-1,
                  training_numPoints=5000,trainingSetSize=50000,embedding_batchSize=30000
                 )
    ```

### Create Brady Movies

12. ```
    output_dir, groups, connections, new_h5s, vidnames = create_brady_videos(
    project_path,watershed_path=f'{project_path}/Encoded_SAE_Euc_Dist/TSNE/zVals_wShed_groups.mat',
    autoencoder_data_path=f'{project_path}/autoencoder_data/',
    video_path=f'{project_path}/videos/')
    ```
    
13. ```
    for region in range(10):
    center_video(region, output_dir, groups, connections, h5s=new_h5s, 
    vidnames=vidnames, animal_fps=25, subs=4, num_pad=250)
    break
    ```
   