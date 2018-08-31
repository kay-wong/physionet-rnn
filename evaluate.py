#!/usr/bin/python3
import tensorflow as tf
import numpy as np
import time, os, glob, argparse
from sklearn.metrics import f1_score, recall_score

# User-defined
from network import Network
from diagnostics import Diagnostics
from data import Data
from model import Model
from config import config_test, directories

import pandas as pd
tf.logging.set_verbosity(tf.logging.ERROR)

def evaluate(config, directories, ckpt, args):
    pin_cpu = tf.ConfigProto(allow_soft_placement=True, device_count = {'GPU':0})
    start = time.time()
    #eval_record_paths = glob.glob('{}/*.record'.format(directories.eval))
    #eval_tokens, eval_labels = Data.load_data_tfrecords(eval_record_paths)
    
        

    # Build graph
    model = Model(config, directories, args=args, evaluate=True)
    # Restore the moving average version of the learned variables for eval.
    variables_to_restore = model.ema.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    with tf.Session(config=pin_cpu) as sess:
        # Initialize variables
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())
        #assert (ckpt.model_checkpoint_path), 'Missing checkpoint file!'

        if args.restore_last and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Most recent {} restored.'.format(ckpt.model_checkpoint_path))
        else:
            if args.restore_path:
                new_saver = tf.train.import_meta_graph('{}.meta'.format(args.restore_path))
                new_saver.restore(sess, args.restore_path)
                print('Previous checkpoint {} restored.'.format(args.restore_path))

        #eval_dict = {model.training_phase: False, model.example: eval_tokens, model.labels: eval_labels}

        #y_pred, v_acc = sess.run([model.pred,model.accuracy], feed_dict=eval_dict)
        #v_f1 = f1_score(eval_labels, y_pred, average='macro', labels=np.unique(y_pred))
        #test_handle = sess.run(model.test_iterator.string_handle())
        #eval_handle = sess.run(model.eval_iterator.string_handle())
        feed_dict_eval = {model.training_phase: False} 
        sess.run(model.eval_iterator.initializer)
        v_acc, y_true, y_pred = sess.run([model.accuracy, model.labels, model.pred], feed_dict=feed_dict_eval)
        v_f1 = f1_score(y_true, y_pred, average='macro', labels=np.unique(y_pred))
        v_sensitivity = recall_score(y_true, y_pred,  average='weighted', labels=np.unique(y_pred))

        results_preds = pd.DataFrame({'True': y_true, 'Pred':y_pred})
        results_preds.to_csv('PredOutputs_setc.csv')

        print("Validation accuracy/+P: {:.3f}".format(v_acc))
        print("Validation Se: {:.3f}".format(v_sensitivity))
        print("Validation F1: {:.3f}".format(v_f1))
        print("Eval complete. Duration: %g s" %(time.time()-start))

        return v_acc


def main(**kwargs):
    parser = argparse.ArgumentParser()
#     parser.add_argument("-i", "--input", help="path to test dataset in h5 format")
    parser.add_argument("-r", "--restore_path", help="path to model to be restored", type=str)
    parser.add_argument("-rl", "--restore_last", help="restore last saved model", action="store_true")

    args = parser.parse_args()

    # Load training, test data
    ckpt = tf.train.get_checkpoint_state(directories.checkpoints)

    # Evaluate
    val_accuracy = evaluate(config_test, directories, ckpt, args)

if __name__ == '__main__':
    main()
