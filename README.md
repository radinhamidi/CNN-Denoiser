# Image Denoising Using Variational ConvolutionalNeural Network

We propose a novel architecture for image reconstruction by utilizing convolutional neural networks (CNN). The main idea is to use such networks to learn statistical distribution of pixels and extract concepts in form of feature vectors. Most of the image noises are come from additive Gaussian Noise (AWGN). By changing ordinary CNN architecture and also using generative models these additive noise can be detected and deducted from the disrupted images. Using variational neurons as the fundamental elements of the proposed architecture will let us to denoise both low-level and high-level distorted images. Contrary to the existing discriminative denoisers, our proposed model will need significantly less amount of training data to achieve acceptable performance and also it needs less time for calculation. We ran various set of experiments on proposed and competitive models. This conducted evaluations show that our model is efficient and making it practical denoising applications. 


# Contributions

Our contributions can be enumerated as follows:
- We  formally  define  problem  of  image  denoising  andproposed a novel approach based on convolutional neuralnetwork. This neural network will use variational autoen-coders  setup  to  capture  Gaussian  distributions  over  thepixel values.
- We will test our model with wide range of noise levels tostudy robustness of model against different type of noises.
- We  evaluate  our  proposed  model  over  the  well-knownimage dataset to compare the results. We will show thatproposed model will outperform current methods that arebased on other approaches

## Files and Folders Intro
asd

## Dataset files

You can download the dataset by runing `stl10_input.py` file. After running the file, it automatically download the dataset and save it to the local directory. Also there are moudles in the file for single image plot and read labels. Local directories setting can be changed using these variables at the beginning of the file:

- Path to the directory with the data: `DATA_DIR = './data'`
- Url of the binary data: `DATA_URL = 'http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz'`
- Path to the binary train file with image data: `DATA_PATH = './data/stl10_binary/train_X.bin'`
- Path to the binary train file with labels: `LABEL_PATH = './data/stl10_binary/train_y.bin'`

## Train a New Model

All your files and folders are presented as a tree in the file explorer. You can switch from one to another by clicking a file in the tree.

## Trained Models

You can rename the current file by clicking the file name in the navigation bar or by clicking the **Rename** button in the file explorer.

## Results

## Demo 

In the following you can see two sample of denoising process captured in different epoches. They are put in sequence as a gif for better infrerence.

![Single Image Example](https://github.com/radrammer/CNN-Denoiser/blob/master/gifs/single_denoise.gif)

![Group of Images](https://github.com/radrammer/CNN-Denoiser/blob/master/gifs/group_denoise.gif)


# Classification On Denoised Images

We can perform a classification on de-noised images in order to seeimpact of denoising on a classifier performance. Therefore a classifier trained to classifies the CIFAR-10 dataset without noises. Then a noisy version of CIFAR-10 is given to the de-noising model and results are used for the classification. The difference between original images classification results and reconstricted version of noisy ones can depict performance of the denoising model.


## Evaluation

Evaluation results can be found below:

|       CIFAR-10 |Original (Without Noise)		 |Reconstructed The Noisy Input                          |
|----------------|-------------------------------|-----------------------------|
|Accuracy|            |           |
|F-1 Measure         |          |           |


## Acknowledgment
This project is done for EE8204 final project. 
Instructor: Dr. Kandasamy Illanko
by Radin Hamidi Rad
Ryerson University
Summer 2020
