#!/usr/bin/env python3

class config_train(object):
    mode = 'beta'
    n_layers = 5
    num_epochs = 24
    batch_size = 256
    ema_decay = 0.999
    learning_rate = 1e-4
    momentum = 0.9
    vocab_size = 3002
    rnn_layers = 2
    embedding_dim = 100
    rnn_cell = 'gru'
    hidden_units = 256
    output_keep_prob = 0.75
    max_seq_len = 100
    n_features = 86
    recurrent_keep_prob = 0.8
    conv_keep_prob = 0.5
    n_classes = 2
    embedding= 'train'
    #embedding= 'glove_50d'
    #embedding= 'fasttext'

class config_test(object):
    mode = 'alpha'
    n_layers = 5
    num_epochs = 25
    batch_size = 256
    ema_decay = 0.999
    learning_rate = 1e-4
    momentum = 0.9
    vocab_size = 3002
    rnn_layers = 2
    embedding_dim = 100
    rnn_cell = 'gru'
    hidden_units = 256
    output_keep_prob = 0.75
    max_seq_len = 100
    n_features = 86
    recurrent_keep_prob = 1.0
    conv_keep_prob = 1.0
    n_classes = 2

class embedding_config(object):
    pretrained = False
    embedding_dim = 50
    
    
class directories(object):
    train = '../data/tfrecords_tst/train' 
    test = '../data/tfrecords_tst/test'  #'../data/REQ/REQ_tokenized_test.h5'
    eval = '../data/REQ/REQ_tokenized_test.h5'
    vocabulary = '../data/REQ/REQ_vocab.csv'
    embedding = '../data/pretrained_vectors/glove.6B.50d.txt'
    ft_embedding = '../data/pretrained_vectors/REQ_cn_ft.bin'

    tensorboard = 'tensorboard'
    checkpoints = 'checkpoints'
    checkpoints_best = 'checkpoints/best'
    trainlogs = 'trainlogs'
