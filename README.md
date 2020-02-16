# RNN_project

This is a project that simulates image processing of an autonomous car with machine learning. Please check out the [presentation](./RNN-presentation/vortrag.pdf) for further information

The aim is to construct a very simple task of semantic segmentation with just two classes but with time dependency. This means that the training and test data consist of series of images. This time dependency is exploited by a simple RNN leading to a better result than just a CNN that processes single images.

The image data was generated by a small robot following a black line while taking images (two images per second). The semantic segmentation should be done with respect to two classes: lane (black line) and background (floor).

![Robot](/figures/robot.jpg)

A simple recurrent neural network processes the previous segmentation together with the actual image to generate the actual segmentation result.

![RNN architecture](/figures/network_architecture.png)
