# IA-C3D
Inception-Attention 3D Convolutional Network

We combine 3D convolutional kernels, and attention mechanism to propose an approach for weakly-supervised learning action detection. The weakly-supervised learning action detection aims to prediction action locations with video-level labels only. In addition, the IA-C3D framework does not require to generate lots of action proposals in advance, and it therefore saves tremendous computational cost.

## Required Package
- Tensorflow <br/>
- Numpy <br/>
- Scikit-learn <br/>
- Scipy <br/>

## Arguments and Commands

- Training <br/>
usage: main.py [-h] [-m MODE] [-t TRAININGFILE] [-p PATH] [-n NUM]

optional arguments: <br/>
  -h, --help            show this help message and exit <br/>
  -m MODE, --mode MODE  training mode <br/>
  -t TRAININGFILE, --trainingfile TRAININGFILE <br/>
                        data list <br/>
  -p PATH, --path PATH  data path <br/>
  -n NUM, --num NUM     number of classes <br/>

- Test

## C3D pretrained model from 
https://github.com/tqvinhcs/C3D-tensorflow <br/>
