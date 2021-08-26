# Object-Localization

This repository contains the jupyter notebooks to train an Object Detection Convolutional Neural Network. This is a bare-bones training program; i.e. this is not meant to be user-friendly/straight-forward. Please read the **Important Notes** section down below:

## Important Notes

- This repo contains `train.ipynb` and `train_torch.ipynb`. `train.ipynb` is a training program written using **TensorFlow** and has memory leak issues, thus it is abandoned in favour of the **PyTorch** version. Please **do not use `train.ipynb`.**

- This repo contains `dataloader.py` and `rm.py`. This contains a data generator function as well as preprocessing code for the **ImageNet dataset**. It is **strongly recommended** to write your own data generator. Be sure that it `yield (torch.tensor, torch.tensor)`, which corresponds to (X, Y) training pair. By default, the network expects input tensors of shape `(batch_size, 416, 416, 3)`

- In `train_torch.ipynb`, you can go to the **Customize Your Training** section, where you can change the number of classes, hyper-parameters, etc. By default, the number of classes is 1000, corresponding to the 1000 classes from the **ImageNet Dataset**. You should change the number of classes to correspond to whichever dataset you are training on.
