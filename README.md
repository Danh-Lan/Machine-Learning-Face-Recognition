# Deep Learning Project - OT2 2023
## About The Project
This project is a simplified version of ImageNet. The objective is to design a face recognition software based on CNN. Users enter an image containing human faces, then the program returns an annotated image displaying green zones where faces have been detected.

### Built With
- Python 3.11.5
- PyTorch 2.1.0
- CUDA 11.8

## Getting Started
Below are some instructions on setting up this project locally.
### Prerequisites
Make sure that Python version 3.11.5 and PyTorch version 2.1.0 are installed on your machine.
### Installation
1. Clone the repo
   ```sh
   git clone https://github.com/Danh-Lan/OT2-Machine-Learning
   ```
2. Prepare train and test data
   Create two folders ```train_images``` and ```test_images``` under the root directory. In each of them, create one folder named ```0``` containing non-face images and another named ```1``` containing face images.
   Please contact us to get data.

 ## Usage
 ### 1. Train face classifier
 Four available notebooks dedicated to classifiers' training are:
 1. ```01Original.ipynb```: The base notebook. It loads data from ```train_images``` folder, displays images and carries out the training process. You can find an implementation of a very simple neural network architecture with only 2 convolutional and 3 fully-connected layers. 
 2. ```2ImbalancedSampler.ipynb```: A notebook designed to take into account the imbalanced dataset. There are more images labeled face than non-face in the training dataset. We will employ an existing data sampler that corrects this imbalance in binary classification before training. Please refer to [this repository](https://github.com/ufoym/imbalanced-dataset-sampler) for further explanation.
 3. ```3DataAugmentation.ipynb```: A notebook in which we apply data augmentation on the fly, including random resizing, cropping and flipping images.
 4. ```4LRScheduling.ipynb```:  A notebook in which we inject a scheduling mechanism into the model training cycles.

All trained models are saved under the  ```/models``` directory as .pth files. These models will serve for the face detection task.

### 2. Recognise faces within images
The two following notebooks contain the implementation of a face detector that combines four different approaches:
1. ```FaceRecognition.ipynb```: A detector built from sliding window, image pyramid and Non-maxima suppression.
2. ```ImgSegmentation.ipynb```: The input images are segmented before being scanned by the detector implemented in ```FaceRecognition.ipynb```


<!-- LICENSE -->
## License

Distributed under the MIT License.

<!-- CONTACT -->
## Authors
- Thanh Lam DANG - thanh.dang@insa-lyon.fr
- Danh Lan NGUYEN - danh.nguyen1@insa-lyon.fr

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments
This project is part of the Machine Learning & Data Analytics course at INSA Lyon, instructed by Prof. Stefan Duffner.

 
