**Detectron2 Installation and Training for Custom Datasets**

Detectron2 Installation and Training on Custom Datasets for single class

Here's a blog about setting up the environment and installing Detectron2 on a local computer: https://medium.com/@aleenarayamajhi13/detectron2-environment-setup-on-a-local-computer-gpu-enabled-7420a7af14c2

The requirements.txt section has the torch installation command for cuda 11.1, if you have other cuda installed on your computer, get the appropriate installation command from pytorch.org.

Once you have setup the virtual environment and installed requirements.txt, you can install Detectron 2. 

To install Detectron2, you need to clone the official repo from Facebook Research: git clone https://github.com/facebookresearch/detectron2.git

Get inside the detectron2 folder that you have just cloned: cd detectron2

Run: python setup.py build develop

Make sure you have your datasets inside train and val folders inside main folder 'object' and the json files should be renamed as via_region_data.json

Run: python train.py

After the training is completed, run inference.py


References:
1. https://github.com/facebookresearch/detectron2
2. https://github.com/TannerGilbert/Object-Detection-and-Image-Segmentation-with-Detectron2
