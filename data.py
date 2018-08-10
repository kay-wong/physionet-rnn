#!/usr/bin/python3
import tensorflow as tf
from tensorflow.contrib import learn
import numpy as np
import pandas as pd
import re, os, csv
from config import config_train, config_test, directories

class Data(object):

    @staticmethod
    def load_tfrecords(tfrec_paths, batch_size, test=False):

        def _parse_tfrecords(proto_f):
            # Parse each record into tensors
            keys_to_features = {'features':tf.FixedLenFeature((config_train.max_seq_len, config_train.n_features), tf.float32),
                            'y': tf.FixedLenFeature((2), tf.int64)}
            parsed_features = tf.parse_single_example(proto_f, keys_to_features)            
            return parsed_features['features'], parsed_features['y']

        dataset = tf.data.TFRecordDataset(tfrec_paths)
        dataset = dataset.map(_parse_tfrecords)  
        dataset = dataset.shuffle(buffer_size=1024)
        dataset = dataset.batch(batch_size)

        if test:
            dataset = dataset.repeat()  
        
        return dataset

    @staticmethod
    def load_data(filename):
        df = pd.read_hdf(filename, key='df').sample(frac=1).reset_index(drop=True)
        tokens = df['tokens']

        # Get lengths of each row of data
        lens = np.array([len(tokens[i]) for i in range(len(tokens))])

        # Mask of valid places in each row
        mask = np.arange(lens.max()) < lens[:,None]

        # Setup output array and put elements from data into masked positions
        padded_tokens = np.zeros(mask.shape)
        padded_tokens[mask] = np.hstack((tokens[:]))

        return padded_tokens, df['category'].values

    @staticmethod
    def load_dataset(features_placeholder, labels_placeholder, batch_size, test=False):
    
        # def _preprocess(tokens, label):
        #     return tokens, label

        dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
        # dataset = dataset.map(_preprocess)
        dataset = dataset.shuffle(buffer_size=512)

        dataset = dataset.padded_batch(
            batch_size,
            padded_shapes=(tf.TensorShape([None]), tf.TensorShape([])),
            padding_values=(0,0))

        if test:
            dataset = dataset.repeat()

        return dataset
    
    @staticmethod
    def load_vocab_file(fname):
        #load vocab file as dictionary of word to idx mapping
        with open(fname, mode='r') as f:
            reader = csv.reader(f)
            vocabulary = {rows[0]:rows[1] for rows in reader}
        
        return vocabulary

    @staticmethod
    def load_embedding_vectors_glove(vocabulary, glove_filename, vector_size):
        # load embedding_vectors from glove
        embedding_vectors = np.random.uniform(-0.25, 0.25, (len(vocabulary), vector_size))
        #embedding_vector = tf.get_variable('embeddings', [config.vocab_size, config.embedding_dim])

        glove_word2idx = {}
        with open(glove_filename) as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype="float32")

                glove_word2idx[word] = vector
        
        for word, idx in vocabulary.items():
            try:
                glove_vect = glove_word2idx[word]
                embedding_vectors[int(idx)] = glove_vect
            except KeyError:
                #print("{} not found".format(word))
                pass                  
        

        embedding_vectors = tf.cast(embedding_vectors, tf.float32)
        return embedding_vectors

    @staticmethod
    def load_embedding_vectors_fasttext(vocabulary, fasttext_path, vector_size):
        # load embedding_vectors from fasttext wordvec model bin file
        embedding_vectors = np.random.uniform(-0.25, 0.25, (len(vocabulary), vector_size))
        ft_model = FastText.load_fasttext_format(fasttext_path)

        for word, idx in vocabulary.items():
            try:
                embedding_vectors[int(idx)] = ft_model[word]
            except KeyError:
                #print("{} not found".format(word))
                pass
        embedding_vectors = tf.cast(embedding_vectors, tf.float32)
        return embedding_vectors


    