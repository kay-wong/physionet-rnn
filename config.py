#!/usr/bin/env python3

class config_train(object):
    mode = 'beta'
    n_layers = 5
    num_epochs = 24
    batch_size = 128
    ema_decay = 0.999
    learning_rate = 1e-4
    momentum = 0.9
    rnn_layers = 2
    rnn_cell = 'gru'
    hidden_units = 256
    output_keep_prob = 0.75
    max_seq_len = 100
    n_features = 86
    recurrent_keep_prob = 0.8
    conv_keep_prob = 0.5
    n_classes = 2

class config_test(object):
    mode = 'alpha'
    n_layers = 5
    num_epochs = 25
    batch_size = 128
    ema_decay = 0.999
    learning_rate = 1e-4
    momentum = 0.9
    rnn_layers = 2
    rnn_cell = 'gru'
    hidden_units = 256
    output_keep_prob = 0.75
    max_seq_len = 100
    n_features = 86
    recurrent_keep_prob = 1.0
    conv_keep_prob = 1.0
    n_classes = 2

    
class directories(object):
    train = './data/tfrecords/seta/'
    eval = './data/tfrecords/setc'

    tensorboard = 'tensorboard'
    checkpoints = 'checkpoints'
    checkpoints_best = 'checkpoints/best'
    trainlogs = 'trainlogs'
