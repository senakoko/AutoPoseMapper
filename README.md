# AutoPoseMapper

This package is meant to filter pose-estimation from Markerless
deep learning methods like [maDLC](https://github.com/DeepLabCut/DeepLabCut) or 
[SLEAP](https://sleap.ai). In addition, it uses the output from
[idtracker.ai](https://idtrackerai.readthedocs.io/en/latest/#) to fix
instances where there are swapped points. 

This package also has an implementation of a convolutional autoencoder. It helps reduce the dimensions of images 
similar to non-linear PCA and retains most of its information.

AutoPoseMapper does not replace maDLC or SLEAP. It has helper functions to use 
with those packages.

In this current version, the AutoPoseMapper was only tested with two animals.
However, it should work with more than two based on its implementation.

It also inherits the python version of the [Motion
Mapper](https://github.com/bermanlabemory/motionmapperpy) to create behavioral spaces

## Installation:
1. Open a terminal
2. Download the repository <code> git clone https://github.com/senakoko/AutoPoseMapper.git </code>
3. Create a new conda environment <code>conda create -n apm python=3.7 -y </code>
4. Activate desired conda environment <code> conda activate apm </code>
5. Navigate to unzipped folder containing requirements.txt.  
6. Run   
<code>
pip install -r requirements.txt  
</code>   

7. Install tensorflow gpu if you have the hardware  
<code>
pip install tensorflow-gpu==2.4.0
</code>  

## Demo:
Go to the Demo Folder. The Demos show how to use the autoposemapper. You can skip some of them but not
the **Demo_0**.  

Demo_0 - Create Project  
Demo_0a - Add New Videos to Project  
Demo_1a - Use SLEAP helper tools  
Demo_1b - Use DLC helper tools  
Demo_1c - Use Convolutional Autoencoder tools  
Demo_1d - Use ID Tracker.AI tools  
Demo_2 - Use Autoencoder tools  
Demo_3a - Fix Bad Areas  
Demo_3a - Add Body Center  
Demo_4 - Run MotionMapper  
Demo_5 - Make Videos  





