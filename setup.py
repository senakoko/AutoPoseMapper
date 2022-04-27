#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AutoPoseMapper 0.0.1
https://github.com/senagezo/AutoPoseMapper
Licensed under GNU Lesser General Public License v3.0
"""


import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

if __name__ == "__main__":

    setuptools.setup(
        name="autoposemapper",
        version="0.0.1",
        author="Sena Agezo",
        author_email="senaagezo@gmail.com",
        description="Filter pose-estimation from Markerless Deep Learning Trackers",
        long_description=long_description,
        long_description_content_type="text/markdown",
        # url="https://github.com/senaagezo/autoposemapper.git",
        install_requires=[
            "numpy==1.21.6",
            "scikit-image==0.19.2",
            "hdf5storage==0.1.18",
            "scipy==1.7.3",
            "scikit-learn==1.0.2",
            "pandas==1.3.5",
            "matplotlib==3.5.1",
            "moviepy==1.0.3",
            "opencv-python==4.5.5.64",
            "easydict==1.9",
            "tables==3.7.0",
            "pyyaml==6.0",
            "tqdm==4.64.0",
            "jupyter==1.0.0",
            "jupyterlab==3.3.4",
            "umap-learn"
        ],
        packages=setuptools.find_packages(),
        zip_safe=False,
        classifiers=[
            "Intended Audience :: Science/Research",
            "Programming Language :: Python :: 3",
            "Topic :: Scientific/Engineering :: Image Recognition"
        ],
    )
