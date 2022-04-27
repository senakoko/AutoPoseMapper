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
1. (OPTIONAL) Create a new conda environment <code>conda create -n apm python=3.7 -y </code>
2. Activate desired conda environment <code> conda activate apm </code>
3. Download the repository and unzip contents. Open terminal and 
navigate to unzipped folder containing setup.py.
4. Run   
<code>
pip install numpy scikit-image hdf5storage scipy scikit-learn pandas matplotlib moviepy opencv-python easydict tables 
pip install tensorflow-gpu==2.4.0 pyyaml tqdm jupyter jupyterlab  
python setup.py install
</code>  

## Demo:
Go to the Demo Folder. The Demos show how to use the autoposemapper. You can skip some of them but not
the Demo_0.  
Demo_0 - Create Project  
Demo_0a - Add New Videos to Project  
Demo_1a - Use SLEAP helper tools  
Demo_1b - Use DLC helper tools  
Demo_1c - Use Convolutional tools  
Demo_2 - Use Autoencoder tools  
Demo_3 - Use ID Tracker.AI Tools  
Demo_4 - Run MotionMapper  
Demo_5 - Make Videos  





