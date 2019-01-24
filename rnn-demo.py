"""
This module prepares midi file data and feeds it to the neural
network for training
"""
import sys
import json
import yaml
import time
import h5py
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import keras
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import CuDNNLSTM, GRU, CuDNNGRU, Input
from keras import regularizers
from keras.utils import multi_gpu_model
from keras.models import Model
import tensorflow as tf

LayerType = CuDNNGRU

def ohe(matrix, n):
    cube = np.zeros((matrix.shape[0], matrix.shape[1], n), dtype=np.int)

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            cube[row, col, matrix[row, col]] = 1

    return cube

class SequenceTrainer():
    def get_par (self, pars, keys, default):
        try:
            p = pars
            for key in keys:
                p = p[key]
            return p

        except:
            return default

    def prepare_data(self, data, split):
        """
        The original input consists of a list of 5 matrices of sequence
        data (X) and a list of 5 matrices as target (Y)
        X[i].shape = (n, seq length, # of categories (usually 128))
        Y[i].shape = (n, # of categories)

         Args:
             data (list): list of X/Y_train, X/Y_val and X/Y_test
             split (list): List containing training fraction and validation fraction

        Returns:
            Four arrays: X_train, Y_train, X_val, Y_val
        """
        # Create one hot encoded vectors
        train_data = ohe(data[0], 128)
        val_data = ohe(data[1], 128)

        # Be sure that Y follows an X sequence
        X_train = train_data[:, :-1, :]
        Y_train = train_data[:, -1:, :]

        X_val = val_data[:, :-1, :]
        Y_val = val_data[:, -1:, :]

        # Remove 2nd index from Y, which is one
        Y_train = Y_train.reshape((Y_train.shape[0], Y_train.shape[2]))
        Y_val = Y_val.reshape((Y_val.shape[0], Y_val.shape[2]))

        return X_train, X_val, Y_train, Y_val

    def single_input_model(self, X, Y, layers, dropout):
        """ Create a simple input/output network

        This model can be trained to associate one voice with one target.

        Args:
            X (list of arrays): contains input data
            Y (list of arrays): targets
            layers (list): list of two lists of layers to be created. The first
                list contains the sizes of RNN layers to be created;
                the second list the sizes of Dense layers
            dropout (float): dropout value; if > 0 a dropout layer is added
                to each RNN or Dense layer

        Returns:
            The model
        """
        rnn_layers = layers[0]
        dense_layers = layers[1]

        # In this test using the kernel regularizer = weight decay
        l2k = self.l2k # Weights regularizer
        l2a = self.l2a # activity regularizer
        l2r = self.l2r # self.l2r # recurrent regularizer
        print ('*** l2k =', l2k, 'l2a =', l2a, 'l2r =', l2r)

        input_layer = Input(shape=(X.shape[1], X.shape[2]), name='Input_Layer')

        if len(rnn_layers) == 1:
            model = LayerType(rnn_layers[0],
                              kernel_regularizer=regularizers.l2(l2k),
                              recurrent_regularizer=regularizers.l2(l2r),
                              activity_regularizer=regularizers.l2(l2a),
                              name='RNN_1')(input_layer)
        else:
            model = LayerType(rnn_layers[0], return_sequences=True,
                              kernel_regularizer=regularizers.l2(l2k),
                              recurrent_regularizer=regularizers.l2(l2r),
                              activity_regularizer=regularizers.l2(l2a),
                              name='RNN_1')(input_layer)
            for layer in range(1, len(rnn_layers) - 1):
                model = LayerType(rnn_layers[layer],
                              return_sequences=True,
                              kernel_regularizer=regularizers.l2(l2k),
                              recurrent_regularizer=regularizers.l2(l2r),
                              activity_regularizer=regularizers.l2(l2a),
                              name='RNN_' + str(layer+1))(model)
                if dropout > 0:
                    model= Dropout(dropout)(model)

            name = 'RNN_{:d}'.format(len(rnn_layers))
            model = LayerType(rnn_layers[-1],
                              kernel_regularizer=regularizers.l2(l2k),
                              recurrent_regularizer=regularizers.l2(l2r),
                              activity_regularizer=regularizers.l2(l2a),
                              name=name)(model)
            if dropout > 0:
                model= Dropout(dropout)(model)

        for i, layer in enumerate(dense_layers):
            model = Dense(layer, activation='relu',
                          kernel_regularizer=regularizers.l2(l2k),
                          activity_regularizer=regularizers.l2(l2a),
                          name='Dense_'+str(i))(model)
            #model = BatchNormalization()(model)
            if dropout > 0:
                model = Dropout(dropout)(model)

        model = Dense(Y.shape[1], activation='softmax', name='Dense_softmax')(model)

        main_model = Model(inputs=input_layer, outputs=[model])

        return main_model

    def setup_model(self, model_def, X, Y, dropout, gpu):
        """ Sets up a Neural Network to generate music

        Args:
            model_type (string): type of model to set up
            X (array): input sequences
            Y (array): imput target
            layers (list): list containing the layer sizes for the model
            dropout (float): dropout fraction
            gpu (int): when > 1, a multi gpu model will be built

        Returns:
            the created model
        """
        model_type = model_def['model']
        layers = model_def['layers']
        if gpu > 1:
            with tf.device("/cpu:0"):
                model = self.single_input_model(X, Y, layers, dropout)
                model = multi_gpu_model(model, gpus=gpu)
                print('Running a multi GPU model on a', model_type, 'model')
        else:
            with tf.device("/gpu:0"):
                model = self.single_input_model(X, Y, layers, dropout)
                print('Running a single GPU model on a', model_type, 'model')

        model.compile(optimizer=keras.optimizers.Adam (),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        return model

    def train(self, hp, model_def, data, dropout, batch_size,
              epochs, gpu):

        hp = dict(hp)
        split = 0.8
        hp['batch_sizes'] = [batch_size]
        hp['dropouts'] = [dropout]

        X_train, X_val, Y_train, Y_val = self.prepare_data(data, split)
        print('X shape', X_train.shape)
        print('Number of training sequences:', len(X_train))
        print('Number of validation sequences:', len(X_val))
        print('Length of sequences is', X_train.shape[1])

        model = self.setup_model(model_def, X_train, Y_train, dropout, gpu)
        model.summary()

        print('\nStarted training the model')
        print('Batch size:', batch_size)
        print('GPU\'s:', gpu)
        print('Dropout:', dropout)

        history = model.fit(X_train, Y_train,
                       verbose=1,
                       epochs=epochs,
                       batch_size=batch_size,
                       validation_data=(X_val, Y_val))

        return history

    def train_sequence(self, hyper_pars, notes_file):
        self.hyper_pars = hyper_pars

        model_types = self.get_par(hyper_pars, ['models'], None)
        batch_sizes = self.get_par(hyper_pars, ['batch_sizes'], [128])
        dropouts = self.get_par(hyper_pars, ['dropouts'], [0.3])
        epochs = self.get_par(hyper_pars, ['epochs'], 100)
        gpus = self.get_par(hyper_pars, ['gpus'], [1])

        print('Tensorflow version:', tf.__version__)
        print('Keras version:', keras.__version__)

        #data = read_sequences(notes_file)
        #self.stf(data)
        #sys.exit()
        train_data = np.genfromtxt('train.csv', delimiter=',', dtype=np.int)
        val_data = np.genfromtxt('val.csv', delimiter=',', dtype=np.int)

        n_runs = len(model_types) * len(dropouts) * len(batch_sizes) * \
                 len(gpus)
        columns = ['Epochs', 'Model type', 'Dropouts', 'Batch size', 'GPU\'s',
                   'Acc', 'Val. Acc', 'Time']
        run_no = 0
        df = pd.DataFrame(np.zeros((n_runs, len(columns))), columns=columns)
        for gpu in gpus:
            for index in model_types:
                model_def = hyper_pars[index]
                for dropout in dropouts:
                    for batch_size in batch_sizes:
                        print('==>', index, '=', str(model_def))

                        self.l2r = 1e-6
                        self.l2k = 1e-6
                        self.l2a = 0.0
                        df.iloc[run_no]['Epochs'] = epochs
                        df.iloc[run_no]['Model type'] = len(model_def)
                        df.iloc[run_no]['Dropouts'] = dropout
                        df.iloc[run_no]['Batch size'] = batch_size
                        df.iloc[run_no]['GPU\'s'] = gpu

                        model_time = time.time()

                        history = self.train(hyper_pars, model_def,
                                             (train_data, val_data),
                                   dropout, batch_size, epochs, gpu)
                        model_time = time.time() - model_time
                        print('CPU time: {:.0f}'.format(model_time))
                        hist = history.history

                        df.iloc[run_no]['Acc'] = hist['acc'][-1]
                        df.iloc[run_no]['Val. Acc'] = hist['val_acc'][-1]
                        df.iloc[run_no]['Time'] = int(model_time)
                        df.to_csv('results.csv')
                        print(df)

                        run_no += 1
                    # for
                # for
            # for
        # for

        return

## Class: music_trainer ###

def main(argv):
    # System wide constants for MusicData
    seed = 42
    np.random.seed(seed)
    random.seed(seed)

    notes_file = 'notes.h5'
    config_file = 'config.yaml'

    # Read hyperparameters
    with open(config_file) as yaml_data:
        hyper_pars = yaml.load(yaml_data)

    # Initialize CPU time measurement
    seconds = time.time()

    SequenceTrainer().train_sequence(hyper_pars, notes_file)

    seconds = int(time.time() - seconds + 0.5)

    print('\n*** Ready in', seconds, 'seconds.')

if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    main(sys.argv[1:])
