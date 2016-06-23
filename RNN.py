import tensorflow as tf
import numpy as np
from tensorflow.models.rnn import rnn, rnn_cell

class RNN:
    def __init__(s, input_size, hidden_size, num_steps, output_size, alpha=0.01):
        s.input_size = input_size
        s.num_steps = num_steps
        s.hidden_size = hidden_size
        s.x = tf.placeholder("float", [None, num_steps, input_size], name="input_placeholder_x")
        s.istate = tf.placeholder("float", [None, 2*hidden_size], name="cell_estate")
        s.y_ = tf.placeholder("float", [None, output_size], name="input_placeholder_y")
        s.weights = {
            'hidden': tf.Variable(tf.random_normal([input_size, hidden_size])),  # Hidden layer weights
            'out': tf.Variable(tf.random_normal([hidden_size, output_size]))
        }
        s.biases = {
            'hidden': tf.Variable(tf.random_normal([hidden_size])),
            'out': tf.Variable(tf.random_normal([output_size]))
        }
        s.ylogits = s.feed_rnn_cell(s.x, s.istate, s.weights, s.biases)
        s.y = tf.nn.softmax(s.ylogits)
        # s.error_measure = tf.reduce_mean(tf.pow(s.y_ - s.y, 2))
        # s.error_measure = tf.reduce_mean(tf.reduce_mean(-s.y_ * tf.log(s.y)))
        s.error_measure = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(s.ylogits, s.y_))
        s.train = tf.train.AdagradOptimizer(learning_rate=alpha).minimize(s.error_measure)
        s.init = tf.initialize_all_variables()
        s.sess = tf.Session()
        s.sess.run(s.init)
        # Evaluate model
        #correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
        #accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    def feed_rnn_cell(s, _X, _istate, _weights, _biases):
        _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
        # Reshape to prepare input to hidden activation
        _X = tf.reshape(_X, [-1,s.input_size])  # (n_steps*batch_size, n_input)
        # Linear activation
        _X = tf.matmul(_X, _weights['hidden']) + _biases['hidden']

        # Define a lstm cell with tensorflow
        lstm_cell = rnn_cell.BasicLSTMCell(s.hidden_size, forget_bias=1.0)
        # Split data because rnn cell needs a list of inputs for the RNN inner loop
        _X = tf.split(0, s.num_steps, _X)  # n_steps * (batch_size, n_hidden)

        # Get lstm cell output
        outputs, states = rnn.rnn(lstm_cell, _X, initial_state=_istate)

        # Linear activation
        # Get inner loop last output
        return tf.matmul(outputs[-1], _weights['out']) + _biases['out']

    def feed_batch(s, batch, expected_outputs):
        # We have to divide each sequence in its num_steps sized chunks
        whole_batch = np.zeros([len(batch),s.num_steps,s.input_size])
        for ii in range(len(batch)):
            for hh in range(s.num_steps):
                for jj in range(s.input_size):
                    whole_batch[ii,hh,jj] = batch[ii][jj+hh]
        #print whole_batch.shape
        return s.sess.run(s.train, feed_dict={s.x: whole_batch, s.y_: expected_outputs, s.istate: np.zeros((len(expected_outputs), 2*s.hidden_size))})

    def error(s,batch,expected_outputs):
        whole_batch = np.zeros([len(batch),s.num_steps,s.input_size])
        for ii in range(len(batch)):
            for hh in range(s.num_steps):
                for jj in range(s.input_size):
                    whole_batch[ii,hh,jj] = batch[ii][jj+hh]
        return s.sess.run(s.error_measure, feed_dict={s.x: whole_batch, s.y_: expected_outputs, s.istate: np.zeros((len(expected_outputs), 2*s.hidden_size))})


    def categorize(s, data):
        unroled_sample = np.zeros([1,s.num_steps,s.input_size])
        for hh in range(s.num_steps):
            for jj in range(s.input_size):
                unroled_sample[0,hh,jj] = data[jj+hh]
        #print whole_batch.shape
        return s.sess.run(s.y, feed_dict={s.x: unroled_sample, s.istate: np.zeros([1,2*s.hidden_size])})
