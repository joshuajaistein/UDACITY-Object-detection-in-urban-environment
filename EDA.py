import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from utils import get_dataset

dataset = get_dataset("training.tfrecord")

def display_instances(tfrecord):
    '''This function shows an image from the tfrecord with its
       corresponding ground truth bounding boxes and labels.
    '''
    # Variables
    name    = tfrecord['filename']
    img     = tfrecord['image'].numpy()
    img_shape = img.shape
    bboxes   = tfrecord['groundtruth_boxes'].numpy()
    classes = tfrecord['groundtruth_classes'].numpy()

    #Display the information of the tfrecord
    print('#########################################TFrecord Information#########################################')
    print('Name of the TFrecord: {}'.format(tfrecord['filename']))
    print('The shape of the image is: {}'.format(img_shape))
    print('The are {} boxes in the image:'.format(len(bboxes)))
    print('The are {} objects in the image:'.format(len(classes)))

    _, ax = plt.subplots(1,figsize=(20, 10))
    # color mapping of classes
    colormap = {1: [1, 0, 0], 2: [0, 0, 1], 4: [0, 1, 0]}

    for cl, bb in zip(classes, bboxes):
        y1, x1, y2, x2 = bb
        y1 = y1*img_shape[0]
        x1 = x1*img_shape[1]
        y2 = y2*img_shape[0]
        x2 = x2*img_shape[1]
        rec = Rectangle((x1, y1), x2- x1, y2-y1, facecolor='none', edgecolor=colormap[cl])
        ax.add_patch(rec)

    # Plot the image with its corresponding bounding boxes
    imgplot = plt.imshow(img)
    plt.show()

# Displaying images
dataset.shuffle(100)

for batches in dataset.take(10):
    display_instances(batches)
