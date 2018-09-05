# -*- coding: utf-8 -*-
# Diagnostic helper functions for Tensorflow session
import tensorflow as tf
from tensorflow.python.client import device_lib
import numpy as np
import os, time
from sklearn.metrics import f1_score, recall_score, precision_score

class Diagnostics(object):
    
    @staticmethod
    def length(sequence):
        used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
        length = tf.reduce_sum(used, 1)
        length = tf.cast(length, tf.int32)
        return length

    @staticmethod
    def last_relevant(output, length):
        batch_size = tf.shape(output)[0]
        max_length = tf.shape(output)[1]
        out_size = int(output.get_shape()[2])
        index = tf.range(0, batch_size) * max_length + (length - 1)
        flat = tf.reshape(output, [-1, out_size])
        relevant = tf.gather(flat, index)
        return relevant

    @staticmethod
    def get_available_gpus():
        local_device_protos = device_lib.list_local_devices()
        #return local_device_protos
        print('Available GPUs:')
        print([x.name for x in local_device_protos if x.device_type == 'GPU'])

    @staticmethod
    def run_diagnostics(model, config, directories, sess, saver, train_handle,
            test_handle, start_time, v_s1_best, epoch, name):
        t0 = time.time()
        improved = ''
        sess.run(tf.local_variables_initializer())
        feed_dict_train = {model.training_phase: False, model.handle: train_handle}
        feed_dict_test = {model.training_phase: False, model.handle: test_handle}

        try:
            t_acc, t_loss, t_summary = sess.run([model.accuracy, model.cost, model.merge_op], feed_dict=feed_dict_train)
            model.train_writer.add_summary(t_summary)
        except tf.errors.OutOfRangeError:
            t_loss, t_acc = float('nan'), float('nan')

        v_acc, v_loss, v_summary, y_true, y_pred = sess.run([model.accuracy, model.cost, model.merge_op, model.labels, model.pred], feed_dict=feed_dict_test)
        model.test_writer.add_summary(v_summary)
        print(y_pred)
        print(y_true)
        v_f1 = f1_score(y_true, y_pred, average='macro', labels=np.unique(y_pred))
        v_sensitivity = recall_score(y_true, y_pred,  average='weighted', labels=np.unique(y_pred))
        if v_sensitivity==1:
            v_sensitivity=0
        v_s1 = min(v_acc, v_sensitivity)

        if v_s1 > v_s1_best:
            v_s1_best = v_s1
            improved = '[*]'
            if epoch>5:
                save_path = saver.save(sess,
                            os.path.join(directories.checkpoints_best, '{0}/rnn_{0}_epoch{1}.ckpt'.format(name, epoch)),
                            global_step=epoch)
                print('Graph saved to file: {}'.format(save_path))

        if epoch % 10 == 0 and epoch>10:
            save_path = saver.save(sess, os.path.join(directories.checkpoints, '{0}/rnn_{0}_epoch{1}.ckpt'.format(name, epoch)), global_step=epoch)
            print('Graph saved to file: {}'.format(save_path))

        msg = 'Epoch {} | S1: {:.3f} | Training Acc: {:.3f} | Test Acc/+P: {:.3f} | Test F1: {:.3f} | Test Se: {:.3f}| Train Loss: {:.3f} | Test Loss: {:.3f} | Rate: {} examples/s ({:.2f} s) {}'.format(epoch, v_s1, t_acc, v_acc, v_f1, v_sensitivity, t_loss, v_loss, int(config.batch_size/(time.time()-t0)), time.time() - start_time, improved)
        print(msg)
        with open(os.path.join(directories.trainlogs, '{}.txt'.format(name)), 'a') as f:
            f.write(msg)
            f.write('\n')
        return v_s1_best

    
    @staticmethod
    def print_x(x):
        print(x)