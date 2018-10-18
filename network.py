""" Network wiring """

import tensorflow as tf
import numpy as np
import glob, time, os
from diagnostics import Diagnostics

class Network(object):

    @staticmethod
    def birnn_dynamic(x, config, training, attention=False):
         # reshape outputs to [batch_size, max_time_steps, n_features]
        #max_time = tf.shape(x)[1]
        #rnn_inputs = tf.reshape(x, [-1, max_time, config.embedding_dim])
        #rnn_inputs = x
        rnn_inputs = tf.reshape(x, [-1, config.max_seq_len, config.n_features])

        sequence_lengths = Diagnostics.length(rnn_inputs)
        init = tf.contrib.layers.xavier_initializer()

         # Choose rnn cell type
        if config.rnn_cell == 'lstm':
            args = {'num_units': config.hidden_units, 'forget_bias': 1.0, 'state_is_tuple': True}
            base_cell = tf.nn.rnn_cell.LSTMCell
        elif config.rnn_cell == 'gru':
            args = {'num_units': config.hidden_units}
            base_cell = tf.nn.rnn_cell.GRUCell
        elif config.rnn_cell == 'layer_norm':
            args = {'num_units': config.hidden_units, 'forget_bias': 1.0, 'dropout_keep_prob': config.recurrent_keep_prob}
            base_cell = tf.contrib.rnn.LayerNormBasicLSTMCell
     
        cell = base_cell

        if config.output_keep_prob < 1:
            # rnn_inputs = tf.nn.dropout(rnn_inputs, self.keep_prob)
            fwd_cells = [tf.nn.rnn_cell.DropoutWrapper(
                cell(**args), 
                output_keep_prob=config.output_keep_prob,
                variational_recurrent=True, dtype=tf.float32) for _ in range(config.rnn_layers)]
            bwd_cells = [tf.nn.rnn_cell.DropoutWrapper(
                cell(**args),
                output_keep_prob=config.output_keep_prob,
                variational_recurrent=True, dtype=tf.float32) for _ in range(config.rnn_layers)]
        else:
            fwd_cells = [cell(**args) for _ in range(config.rnn_layers)]
            bwd_cells = [cell(**args) for _ in range(config.rnn_layers)]

        fwd_cells = tf.contrib.rnn.MultiRNNCell(fwd_cells)
        bwd_cells = tf.contrib.rnn.MultiRNNCell(bwd_cells)
        
        outputs, output_states = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=fwd_cells,
            cell_bw=bwd_cells,
            inputs=rnn_inputs,
            sequence_length=sequence_lengths,
            dtype=tf.float32,
            parallel_iterations=128)

        birnn_output = tf.concat(outputs,2)

        if attention:  # invoke soft attention mechanism - attend to different particles
            summary_vector = attention_A(birnn_output, config.attention_dim, my_method=False)
        else:  # Select last relevant output
            summary_vector = Diagnostics.last_relevant(birnn_output, sequence_lengths)
        
        print(summary_vector.get_shape().as_list())
        # Fully connected layer for classification
        with tf.variable_scope("fc"):
            logits_RNN = tf.layers.dense(summary_vector, units=config.n_classes, kernel_initializer=init)
        
        return logits_RNN
