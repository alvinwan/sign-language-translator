# How to Build a Neural Network to Translate Sign Language into English
Real-Time Sign Language Translation using Computer Vision

This repository includes all source code for the (soon-to-be) tutorial on DigitalOcean with the same title, including:
- A real-time sign language translator based on a live feed.
- Utilities used for portions of the tutorial, such as dataloaders.
- Simple convolutional neural network written in [PyTorch](http://pytorch.org), with pretrained model.

created by [Alvin Wan](http://alvinwan.com), November 2019

<img width="832" alt="Screen Shot 2019-11-29 at 4 42 59 AM" src="https://user-images.githubusercontent.com/2068077/69869958-2c266f00-1263-11ea-9dad-d5f72b56d047.png">
<img width="832" alt="Screen Shot 2019-11-29 at 4 44 34 AM" src="https://user-images.githubusercontent.com/2068077/69869959-2c266f00-1263-11ea-9aab-8af38c1a0946.png">

# Getting Started

For complete step-by-step instructions, see the (soon-to-be) tutorial on DigitalOcean. This codebase was developed and tested using `Python 3.6`. If you're familiar with Python, then see the below to skip the tutorial and get started quickly:

> (Optional) [Setup a Python virtual environment](https://www.digitalocean.com/community/tutorials/common-python-tools-using-virtualenv-installing-with-pip-and-managing-packages#a-thorough-virtualenv-how-to) with Python 3.6.

1. Install all Python dependencies.

```
pip install -r requirements.txt
```

2. Navigate into `src`.

```
cd src
```

3. Launch the script for a sign language translator:

```
python step_5_camera.py
```

# How it Works

See the below resources for explanations of related concepts:

- ["Understanding Least Squares"](http://alvinwan.com/understanding-least-squares/)
- ["Understanding Neural Networks"](http://alvinwan.com/understanding-neural-networks/)

## Acknowledgements

These models are trained on a Sign Language MNIST dataset curated by `tecperson`, as published on [Kaggle](https://www.kaggle.com/datamunge/sign-language-mnist).
