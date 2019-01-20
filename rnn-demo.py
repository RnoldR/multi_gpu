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
import keras
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import CuDNNLSTM, GRU, CuDNNGRU, Input
from keras import regularizers
from keras.utils import multi_gpu_model
from keras.models import Model
import tensorflow as tf

LayerType = CuDNNLSTM

def read_sequences(filename, x_only=False):
    idx_train, idx_val, idx_test = None, None, None
    with h5py.File(filename, mode='r') as hdf5_file:
        X_train = list(hdf5_file["X_train"][:])
        idx_train = hdf5_file['X_train'].attrs['Index']

        try:
            X_val = list(hdf5_file["X_val"][:])
            idx_val = hdf5_file['X_val'].attrs['Index']
        except:
            X_val = None

        try:
            X_test = list(hdf5_file["X_test"][:])
            idx_train = hdf5_file['X_test'].attrs['Index']
        except:
            X_test = None

    data = [X_train, X_val, X_test], [idx_train, idx_val, idx_test]

    return data

def create_y_from_x(X):
    if X is not None:
        if isinstance(X, list):
            Y = [x[:, -1:, :] for x in X]
            X = [x[:, :-1, :] for x in X]
        else:
            Y = X[:, -1:, :]
            X = X[:, :-1, :]

        return X, Y

    return None, None

def reshape_y(Y):
    if Y is not None:
        if isinstance(Y, list):
            Y = [y.reshape((y.shape[0], y.shape[2])) for y in Y]
        else:
            Y = Y.reshape((Y.shape[0], Y.shape[2]))

        return Y

    return None

def ohe(matrix, n):
    cube = np.zeros((matrix.shape[0], matrix.shape[1], n))

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            cube[row, col, matrix[row, col]] = 1

    return cube

def ohe_data(data):
    results = [ohe(matrix, 128) for matrix in data]

    return results

class MusicTrainer():
    def get_par (self, pars, keys, default):
        try:
            p = pars
            for key in keys:
                p = p[key]
            return p

        except:
            return default

    def get_notes(self, note_file, note_saves, selection, verbose, min_length):
        with open(note_file, 'r') as infile:
            preparations = json.load(infile)

        pitches = []
        durations = []

        for piece, features in preparations.items():
            if verbose != 0: print(piece)
            for feature, voices in features.items():
                if verbose != 0: print ('  +', feature)
                for voice, values in voices.items():
                    if voice in selection or selection == ['*']:
                        if feature == 'pitches' and len(values) > min_length:
                            pitches.append(values)
                        elif feature == 'durations'and len(values) > min_length:
                            durations.append(values)

                        flag = '*'
                    else:
                        flag = ''

                    if verbose != 0: print ('    -', voice, flag)

        return pitches, durations

    def prepare_data(self, model_def, data, split):
        """
        The original input consists of a list of 5 matrices of sequence
        data (X) and a list of 5 matrices as target (Y)
        X[i].shape = (n, seq length, # of categories (usually 128))
        Y[i].shape = (n, # of categories)

         Args:
             model_def (string): definition of the model
             data (list): list of X/Y_train, X/Y_val and X/Y_test
             split (list): List containing training fraction and validation fraction

        Returns:
            Four arrays: X_train, Y_train, X_val, Y_val
        """
        model_type = model_def['model']
        X_train, X_val = None, None
        X_train = ohe_data(data[0])
        if data[1] is not None:
            X_val = ohe_data(data[1])

        # Be sure that Y follows an X sequence
        X_train, Y_train = create_y_from_x(X_train)
        X_val, Y_val = create_y_from_x(X_val)

        # Remove 2nd index from Y, which is one
        Y_train = reshape_y(Y_train)
        Y_val = reshape_y(Y_val)

        if model_type == 'single':
            index = 0
            X_train, Y_train = [X_train[index]], [Y_train[index]]
            X_val, Y_val = [X_val[index]], [Y_val[index]]
            print('*** Training on:', index)
        elif (model_type == 'pipeline'):
            # input and output already provided for
            pass
        else:
            # some error occurred, return None
            raise ValueError('Unknown model type: ' + model_type)

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
        rnn_layers = [512, 512, 512, 512] #layers[0]
        dense_layers = [512] #layers[1]

        # In this test using the kernel regularizer = weight decay
        l2k = self.l2k # Weights regularizer
        l2a = self.l2a # activity regularizer
        l2r = self.l2r # self.l2r # recurrent regularizer
        print ('*** l2k =', l2k, 'l2a =', l2a, 'l2r =', l2r)

        input_layer = Input(shape=(X[0].shape[1], X[0].shape[2]), name='Input_Layer')

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

        model = Dense(Y[0].shape[1], activation='softmax', name='Dense_softmax')(model)

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
                model = self.single_input_model(model_type, X, Y, layers, dropout)
                print('Running a single GPU model on a', model_type, 'model')

        model.compile(optimizer=keras.optimizers.Adam (),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        return model

    def train(self, hp, model_def, data, indices, dropout, batch_size,
              epochs, gpu):

        hp = dict(hp)
        split = self.get_par(hp, ['validation_proportion'], 0.25)
        hp['batch_sizes'] = [batch_size]
        hp['dropouts'] = [dropout]

        X_train, X_val, Y_train, Y_val = self.prepare_data(model_def, data, split)
        print('Number of training sequences:', len(X_train[0]))
        print('Number of validation sequences:', len(X_val[0]))
        print('Length of sequences is', data[0][0].shape[1])

        model = self.setup_model(model_def, X_train, Y_train, dropout, gpu)

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

    def train_music(self, hyper_pars, notes_file):
        self.hyper_pars = hyper_pars

        model_types = self.get_par(hyper_pars, ['models'], None)
        batch_sizes = self.get_par(hyper_pars, ['batch_sizes'], [128])
        dropouts = self.get_par(hyper_pars, ['dropouts'], [0.3])
        epochs = self.get_par(hyper_pars, ['epochs'], 100)
        gpus = self.get_par(hyper_pars, ['gpus'], [1])

        print('Tensorflow version:', tf.__version__)
        print('Keras version:', keras.__version__)
        data, indices = read_sequences(notes_file)
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

                        history = self.train(hyper_pars, model_def, data, indices,
                                   dropout, batch_size, epochs, gpu)
                        model_time = time.time() - model_time
                        print('CPU time: {:.0f}'.format(model_time))
                        hist = history.history

                        df.iloc[run_no]['Acc'] = hist['acc'][-1]
                        df.iloc[run_no]['Val. Acc'] = hist['val_acc'][-1]
                        df.iloc[run_no]['Time'] = int(model_time)

                        run_no += 1
                    # for
                # for
            # for
        # for

        print(df)
        df.to_csv('results.csv')

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

    MusicTrainer().train_music(hyper_pars, notes_file)

    seconds = int(time.time() - seconds + 0.5)

    print('\n*** Ready in', seconds, 'seconds.')

if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    np.set_printoptions(threshold=np.nan)
    main(sys.argv[1:])
