import argparse
import glob
import os
import random

import numpy as np

from utils import get_module_logger

# +
import shutil

def split(data_dir):
    """
    Create three splits from the processed records. The files should be moved to new folders in the 
    same directory. This folder should be named train, val and test.

    args:
        - data_dir [str]: data directory, /mnt/data
    """
    
    # TODO: Implement function
    processed_data = os.listdir(data_dir + '/processed')
    random.shuffle(processed_data)

    # variables
    num_tfrecords  = len(processed_data)
    num_training   = num_tfrecords*0.75
    num_validation = num_tfrecords*0.15
    num_testing    = num_tfrecords*0.10
    
    #iterate through all tfrecords 
    for i,tfrecord in enumerate(processed_data):
        # Number of Tfrecords for training
        if i < num_training:
            shutil.move(data_dir+'/processed/'+tfrecord, data_dir + '/train')
          
        # Number of Tfrecords for validation
        elif i >= num_training and i < num_training+num_validation:
            shutil.move(data_dir+'/processed/'+tfrecord, data_dir + '/val')
         
        # Number of Tfrecords for testing
        else:
            shutil.move(data_dir+'/processed/'+tfrecord, data_dir + '/test')
    
    #Provice information about the splits created
    print('The {} tfrecords were splitted as follows: Training: {} | Validation {} | Testing {}'
          .format(num_tfrecords,num_training,num_validation,num_testing))


# -

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Split data into training / validation / testing')
    parser.add_argument('--data_dir', required=True,
                        help='data directory')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    split(args.data_dir)
