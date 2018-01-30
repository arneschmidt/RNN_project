import matplotlib.pyplot as plt
import numpy as np


def show_image(image):
    arr = np.asarray(image)
    plt.imshow(arr, cmap='gray', vmin = 0, vmax = 255)
    plt.pause(.001)
    plt.show(block=False)

def show_images(images):
    for image_i in range(len(images)):
        images[image_i] = np.pad(images[image_i], pad_width=5, mode='constant', constant_values=(0., 0.))
    images = np.concatenate(images, axis=1)
    plt.figure(1)
    show_image(images)

def show_activations(layer_tensors, number_of_filters):
    image = np.ones([70*len(layer_tensors),90*number_of_filters])*255
    for tensor_i in range(len(layer_tensors)):
        tensor = layer_tensors[tensor_i][0]
        tensor = np.rollaxis(tensor, 2)
        for activation_i in range(len(tensor)):
            activation = np.pad(tensor[activation_i], pad_width=5, mode='constant', constant_values=(0.,0.))
            image[70*tensor_i:70*tensor_i+70, 90*activation_i:90*activation_i+90] = activation

    plt.figure(2)
    show_image(image)

def show_inference(image1, image2):
    images = np.concatenate([image1, image2], axis=0)
    plt.figure(3)
    show_image(images)