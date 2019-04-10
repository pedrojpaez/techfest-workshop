from __future__ import absolute_import, print_function

# This code has been adapted from the code used in headline-classifier-local.ipynb notebook to build and train a 1D CNN using Keras with an MXNet backend. You can see the instructions to do this : https://github.com/aws/sagemaker-containers#list-of-provided-environment-variables-by-sagemaker-containers.

# We first import the required libraries. If a library is missing in the container build you can pass a requirements.txt file to do additional pip installs.

from sagemaker_mxnet_container.training_utils import save
import json
import os
import mxnet as mx
from mxnet import gluon, nd, ndarray
import argparse
import numpy as np
import sys
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.models import Sequential, save_mxnet_model
from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.embeddings import Embedding

# We will use gpu or cpu context depending on the instance used.
ctx = mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()

# Create helper functions to obtain the data we passed to the container via S3 channels. Here we have created three helper functions to obtain the train, test and embedding_matrix numpy files.

def load_training_data(base_dir):
    X_train = np.load(os.path.join(base_dir, 'train_X.npy'))
    y_train = np.load(os.path.join(base_dir, 'train_Y.npy'))
    return X_train, y_train

def load_testing_data(base_dir):
    X_test = np.load(os.path.join(base_dir, 'test_X.npy'))
    y_test = np.load(os.path.join(base_dir, 'test_Y.npy'))
    return X_test, y_test

def load_embeddings(base_dir):
    embedding_matrix = np.load( os.path.join(base_dir, 'docs-embedding-matrix.npy'))
    return embedding_matrix
# Acquire hyperparameters and directory locations passed by SageMaker
def parse_args():
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--vocab_size', type=int, default=300)
    parser.add_argument('--num_classes', type=int, default=4)
    
    # Data, model, and output directories
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])
    parser.add_argument('--embeddings', type=str, default=os.environ['SM_CHANNEL_EMBEDDINGS'])
    
    return parser.parse_known_args()

if __name__ == "__main__":
    
    # The main function of this script is the code that will be running. You will se that after obtaining the required data this is directly copied from the headline-classifier-local.ipynb notebook.
    args, unknown = parse_args()
    
    print(args)

    x_train, y_train = load_training_data(args.train)
    x_test, y_test = load_testing_data(args.test)
    embedding_matrix=load_embeddings(args.embeddings)
    
    model = Sequential()
    model.add(Embedding(args.vocab_size, 100, 
                            weights=[embedding_matrix],
                            input_length=40, 
                            trainable=False, 
                            name="embed"))
    model.add(Conv1D(filters=128, 
                         kernel_size=3, 
                         activation='relu',
                         name="conv_1"))
    model.add(MaxPooling1D(pool_size=5,
                               name="maxpool_1"))
    model.add(Flatten(name="flat_1"))
    model.add(Dropout(0.3,
                         name="dropout_1"))
    model.add(Dense(128, 
                        activation='relu',
                        name="dense_1"))
    model.add(Dense(args.num_classes,
                        activation='softmax',
                        name="out_1"))

        # compile the model
    model.compile(optimizer='rmsprop',
                      loss='binary_crossentropy',
                      metrics=['acc'])


    model.summary()
    


    model.fit(x_train, y_train, batch_size=16, epochs=args.epochs, verbose=2)
    model.evaluate(x_test, y_test, verbose=2)
       
    #Finally anything we save within the model directory will be pushed to the S3 output bucket by Sagemaker. The ouput we decide to package is very flexible. In this case we are outputing information necessary for deploying our model.
    model_prefix = os.path.join(args.model_dir, 'model')
    model.save(model_prefix+'.hd5')
    data_name, data_shapes = save_mxnet_model(model=model, prefix=model_prefix,
                                                           epoch=0)
    signature = [{'name': data_name[0], 'shape': [dim for dim in data_desc.shape]}
                 for data_desc in data_shapes]
    with open(os.path.join(args.model_dir, 'model-shapes.json'), 'w') as f:
        json.dump(signature, f)
    
    