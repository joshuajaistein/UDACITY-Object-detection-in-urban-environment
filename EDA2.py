from collections import Counter

def num_objects(path_tfrecord):
    '''This function counts the number of classes that are present in a tfrecord
       and returns a dic of the classes values
    '''
    # Read the information from a single tfrecord
    raw_dataset = tf.data.TFRecordDataset(path_tfrecord)

    # Initialize Counter of Classes to zero
    count_classes = { 1: 0, 2: 0,  4: 0 }

    # Iterate to each element of the tfrecord
    for i, raw_record in enumerate(raw_dataset):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        # Iterate to each feature item from the tfrecord
        for k, v in example.features.feature.items():
            # extract the information from the class labels
            if k == 'image/object/class/label':
                values = v.int64_list.value
                count_classes= Counter(count_classes)+ Counter(values)
        frames = i
    return count_classes,frames
import os

# Store the names of all the tfrecord files
entries = os.listdir('processedRecords')

# Variable to store the total number of classes
num_classes = {1: 0, 2: 0,  4: 0}
num_frames = 0

# Read each TFrecord file and count the number of classes it contains
for name_file in entries:
    count,frames = num_objects('/processedRecords/' + name_file)
    # Update the variables
    num_classes  = Counter(num_classes)+ Counter(count)
    num_frames = num_frames + frames

# Relevant information about the dataset
print('The number of classes found in the {} tfrecords is: {} vehicles | {} pedestrians | {} cyclists'
      .format(len(entries),num_classes[1],num_classes[2],num_classes[4]))
print('The number of images contained in the tfrecors is: {}'.format(num_frames))

import matplotlib.pyplot as plt

# Mapping the num
mapping = {'vehicle': num_classes[1], 'pedestrian': num_classes[2], 'cyclist': num_classes[4]}

plt.figure(figsize=(15,8))
plt.bar(mapping.keys(), mapping.values(), color='b')
plt.show()
