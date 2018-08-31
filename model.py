#!/usr/bin/python3
import tensorflow as tf
import numpy as np
import glob, time, os

from network import Network
from data import Data
from sklearn.model_selection import train_test_split
from diagnostics import Diagnostics

class Model():
    def __init__(self, config, directories, args, evaluate=False):
        # Build the computational graph

        arch = Network.birnn_dynamic

        self.global_step = tf.Variable(0, trainable=False)
        self.handle = tf.placeholder(tf.string, shape=[])
        self.training_phase = tf.placeholder(tf.bool)
        self.rnn_keep_prob = tf.placeholder(tf.float32)


        self.tokens_placeholder = tf.placeholder(tf.float32, [config.max_seq_len, config.n_features])
        self.labels_placeholder = tf.placeholder(tf.int64, None)
        self.test_tokens_placeholder = tf.placeholder(tf.float32)
        self.test_labels_placeholder = tf.placeholder(tf.int64)
        
        # Balanced test distribution
        CLASS0_DIR = '../data/tfrecords/class0/'
        CLASS1_DIR = '../data/tfrecords/class1/'
        c0_paths = glob.glob(CLASS0_DIR+'/*.record')
        c1_paths = glob.glob(CLASS1_DIR+'/*.record')
        mid = len(c1_paths)//2

        train_record_paths, test_record_paths = train_test_split(c0_paths[:1000], train_size = 0.9, random_state=42)
        train_record_paths.extend(c1_paths[:mid])
        test_record_paths.extend(c1_paths[mid:])
        #train_record_paths = glob.glob('{}/*.record'.format(directories.train))
        #test_record_paths = glob.glob('{}/*.record'.format(directories.test))
        steps_per_epoch = len(train_record_paths)//config.batch_size
        

        train_dataset = Data.load_dataset_tfrecords(train_record_paths, config.batch_size)
        test_dataset = Data.load_dataset_tfrecords(test_record_paths, config.batch_size, test=True)
        self.iterator = tf.data.Iterator.from_string_handle(self.handle,
                                                                    train_dataset.output_types,
                                                                    train_dataset.output_shapes)

        self.train_iterator = train_dataset.make_initializable_iterator()
        self.test_iterator = test_dataset.make_initializable_iterator()

        #import pdb; pdb.set_trace()
                
        if evaluate:
            #eval_record_paths = tf.placeholder(tf.string, shape=[])
            eval_record_paths = glob.glob('{}/*.record'.format(directories.eval))
            eval_dataset = Data.load_dataset_tfrecords(eval_record_paths, len(eval_record_paths)-1, test=True)
            self.eval_iterator = eval_dataset.make_initializable_iterator()
            self.example, self.labels = self.eval_iterator.get_next()

            self.logits = arch(self.example, config, self.training_phase)
            self.softmax, self.pred = tf.nn.softmax(self.logits), tf.argmax(self.logits, 1)
            self.ema = tf.train.ExponentialMovingAverage(decay=config.ema_decay, num_updates=self.global_step)
            correct_prediction = tf.equal(self.labels, tf.cast(self.pred, tf.int64))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            return

        else:
            self.example, self.labels = self.iterator.get_next()

        self.logits = arch(self.example, config, self.training_phase)
        self.softmax, self.pred = tf.nn.softmax(self.logits), tf.argmax(self.logits, 1)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        #self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels)
        #self.scaled_error = tf.multiply(self.cross_entropy, class_weight)
        #self.cost = tf.reduce_mean(self.scaled_error)
        hm = tf.reduce_max(self.logits, axis=1)
        #Diagnostics.print_x(hm)
        #print(self.logits)
        #print(self.pred)
        self.cross_entropy = tf.nn.weighted_cross_entropy_with_logits(targets=tf.cast(self.labels, tf.float32), logits=hm, pos_weight=1.2)
        
        self.cost = tf.reduce_mean(self.cross_entropy)

        #epoch_bounds = [64, 128, 256, 420, 512, 720, 1024]
        #epoch_bounds = [1,3,4,5,6,7,8]
        epoch_bounds = [1,5,6,7,9,11,12]
        lr_values = [1e-3, 4e-4, 1e-4, 6e-5, 1e-5, 6e-6, 1e-6, 2e-7]

        learning_rate = tf.train.piecewise_constant(self.global_step, boundaries=[s*steps_per_epoch for s in
            epoch_bounds], values=lr_values)

        with tf.control_dependencies(update_ops):
            # Ensures that we execute the update_ops before performing the train_step
            if args.optimizer=='adam':
                self.opt_op = tf.train.AdamOptimizer(learning_rate).minimize(self.cost, global_step=self.global_step)
            elif args.optimizer=='momentum':
                self.opt_op = tf.train.MomentumOptimizer(learning_rate, config.momentum,
                    use_nesterov=True).minimize(self.cost, global_step=self.global_step)

        self.ema = tf.train.ExponentialMovingAverage(decay=config.ema_decay, num_updates=self.global_step)
        maintain_averages_op = self.ema.apply(tf.trainable_variables())

        with tf.control_dependencies(update_ops+[self.opt_op]):
            self.train_op = tf.group(maintain_averages_op)

        self.str_accuracy, self.update_accuracy = tf.metrics.accuracy(self.labels, self.pred)
        precision, self.update_precision = tf.metrics.precision(self.labels, self.pred)
        #recall, self.update_recall = tf.metrics.recall(self.labels, self.pred)
        recall, self.update_recall = tf.metrics.recall(self.labels, self.pred)

        #self.f1 = 2 * precision * recall / (precision + recall)
        self.f1 = 2 * precision * recall / (precision + recall)
        self.precision = precision
        self.recall = recall
        correct_prediction = tf.equal(self.labels, tf.cast(self.pred, tf.int64))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        tf.summary.scalar('accuracy', self.accuracy)
        tf.summary.scalar('learning_rate', learning_rate)
        tf.summary.scalar('cost', self.cost)
        tf.summary.scalar('f1 score', self.f1)
        self.merge_op = tf.summary.merge_all()

        self.train_writer = tf.summary.FileWriter(
            os.path.join(directories.tensorboard, '{0}/{0}_train_{1}'.format(args.name, time.strftime('%d-%m_%I:%M'))), graph=tf.get_default_graph())
        self.test_writer = tf.summary.FileWriter(
            os.path.join(directories.tensorboard, '{0}/{0}_test_{1}'.format(args.name, time.strftime('%d-%m_%I:%M'))))


